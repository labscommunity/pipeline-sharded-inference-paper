"""Micro-batching: 2 streams over the 2-node 2-stage v5_beam pipeline.

Uses threading + a lock on the shared socket to charlie. Each thread runs
one stream's decode loop. While thread A is inside stage 0 inference or
inside a socket recv, thread B can be using the other resource. This is
the Discovery #3 mechanism applied to the current v5_beam shards.
"""
import os, socket, struct, sys, time, threading, numpy as np, openvino as ov
from transformers import AutoTokenizer

TARGET_MODEL = r"C:\cascadia\models\llama-3.1-8b-int4"
STAGE0_SHARD = os.environ.get("STAGE0_SHARD", r"C:\cascadia\shards_2stage_v5_beam\stage_0")
STAGE1_HOST = os.environ.get("STAGE1_HOST", "192.168.86.28")
STAGE1_PORT = int(os.environ.get("STAGE1_PORT", "19100"))
PROMPT = os.environ.get("SPEC_PROMPT", "What is the capital of France?")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "128"))
NUM_STREAMS = int(os.environ.get("NUM_STREAMS", "2"))


def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(65536, n - len(buf)))
        if not chunk:
            raise ConnectionError(f"peer closed, got {len(buf)} of {n} bytes")
        buf.extend(chunk)
    return bytes(buf)


class DistributedPipeline:
    """Shared socket to stage 1 worker. Per-stream attention_mask state."""

    def __init__(self, sock, num_streams):
        self.sock = sock
        self.sock_lock = threading.Lock()
        self.num_streams = num_streams
        self.valid_masks = [np.ones(4096, dtype=np.int64) for _ in range(num_streams)]
        self.cache_lens = [0] * num_streams
        self.logical_positions = [0] * num_streams

    def reset(self, stream_id):
        with self.sock_lock:
            self.sock.sendall(struct.pack("<II", stream_id, 0))
            _, _ = struct.unpack("<II", recv_exact(self.sock, 8))
        self.valid_masks[stream_id][:] = 1
        self.cache_lens[stream_id] = 0
        self.logical_positions[stream_id] = 0

    def forward_stage1(self, stream_id, hidden_states, new_token_count):
        """Send to stage 1 worker, receive logits. Thread-safe via socket lock."""
        cache_len = self.cache_lens[stream_id]
        total = cache_len + new_token_count
        mask = self.valid_masks[stream_id]
        if total > len(mask):
            new_size = max(total * 2, len(mask) * 2)
            new_mask = np.ones(new_size, dtype=np.int64)
            new_mask[:len(mask)] = mask
            self.valid_masks[stream_id] = new_mask
            mask = new_mask
        attn_mask = np.empty((1, total), dtype=np.int64)
        attn_mask[0, :cache_len] = mask[:cache_len]
        attn_mask[0, cache_len:] = 1
        hidden = np.ascontiguousarray(hidden_states, dtype=np.float32)

        logical_pos = self.logical_positions[stream_id]
        msg = (
            struct.pack("<IIII", stream_id, 1, logical_pos, total)
            + attn_mask.tobytes()
            + struct.pack("<II", new_token_count, 4096)
            + hidden.tobytes()
        )

        with self.sock_lock:
            self.sock.sendall(msg)
            seq_len, vocab_size = struct.unpack("<II", recv_exact(self.sock, 8))
            logits_bytes = recv_exact(self.sock, seq_len * vocab_size * 4)

        logits = np.frombuffer(logits_bytes, dtype=np.float32).reshape(1, seq_len, vocab_size)
        self.cache_lens[stream_id] = total
        self.logical_positions[stream_id] += new_token_count
        return logits


def stream_decode(stream_id, stage0_req, pipeline, input_ids, max_tokens, results, start_barrier):
    """Decode loop for one stream. Runs on its own thread."""
    s0_valid = np.ones(4096, dtype=np.int64)
    s0_cache_len = 0
    s0_logical = 0

    def s0_attn(new_tokens):
        total = s0_cache_len + new_tokens
        mask = s0_valid
        attn = np.empty((1, total), dtype=np.int64)
        attn[0, :s0_cache_len] = mask[:s0_cache_len]
        attn[0, s0_cache_len:] = 1
        return attn

    def stage0(ids, n):
        nonlocal s0_cache_len, s0_logical
        attn = s0_attn(n)
        pos = np.arange(s0_logical, s0_logical + n, dtype=np.int64).reshape(1, -1)
        stage0_req.infer({"input_ids": ids, "attention_mask": attn,
                          "position_ids": pos, "beam_idx": np.zeros(1, dtype=np.int32)})
        hidden = stage0_req.get_output_tensor(0).data.copy()
        s0_cache_len += n
        s0_logical += n
        return hidden

    stage0_req.reset_state()
    pipeline.reset(stream_id)

    start_barrier.wait()  # synchronize both threads to begin together

    t0 = time.perf_counter()

    # Prefill
    n = input_ids.shape[1]
    hidden = stage0(input_ids, n)
    logits = pipeline.forward_stage1(stream_id, hidden, n)
    nt = int(np.argmax(logits[0, -1, :]))
    gens = [nt]

    for _ in range(1, max_tokens):
        hidden = stage0(np.array([[nt]], dtype=np.int64), 1)
        logits = pipeline.forward_stage1(stream_id, hidden, 1)
        nt = int(np.argmax(logits[0, -1, :]))
        gens.append(nt)

    elapsed = time.perf_counter() - t0
    results[stream_id] = {"gens": gens, "elapsed": elapsed, "tok_s": max_tokens / elapsed}


def main():
    print(f"NUM_STREAMS={NUM_STREAMS}  STAGE1={STAGE1_HOST}:{STAGE1_PORT}", flush=True)
    tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    input_ids = tok.encode(PROMPT, return_tensors="np").astype(np.int64)

    print(f"Loading stage_0 shard: {STAGE0_SHARD}", flush=True)
    core = ov.Core()
    # WORKAROUND: per-stream compile_model (shared compile causes Add shape
    # errors on stateful reset across multiple InferRequests on this OV build)
    stage0_model = core.read_model(os.path.join(STAGE0_SHARD, "openvino_model.xml"))
    stage0_reqs = []
    for s in range(NUM_STREAMS):
        c = core.compile_model(stage0_model, "GPU")
        stage0_reqs.append(c.create_infer_request())
    print(f"Created {NUM_STREAMS} independent stage 0 InferRequests on alpha GPU", flush=True)

    print(f"Connecting to stage 1 worker...", flush=True)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(300.0)
    sock.connect((STAGE1_HOST, STAGE1_PORT))
    pipeline = DistributedPipeline(sock, num_streams=NUM_STREAMS)

    def run_batch():
        results = [None] * NUM_STREAMS
        barrier = threading.Barrier(NUM_STREAMS)
        threads = [threading.Thread(
            target=stream_decode,
            args=(s, stage0_reqs[s], pipeline, input_ids, MAX_TOKENS, results, barrier))
            for s in range(NUM_STREAMS)]

        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        total_elapsed = time.perf_counter() - t0
        return results, total_elapsed

    print("Warmup x2...", flush=True)
    for _ in range(2):
        run_batch()

    print(f"Timed x3 ({NUM_STREAMS} streams × {MAX_TOKENS} tokens each):", flush=True)
    aggregates = []
    for i in range(3):
        results, total_elapsed = run_batch()
        total_tokens = NUM_STREAMS * MAX_TOKENS
        agg_tps = total_tokens / total_elapsed
        per_stream = [r["tok_s"] for r in results]
        aggregates.append(agg_tps)
        print(f"  run {i+1}: total {total_tokens} tok / {total_elapsed:.3f} s = {agg_tps:.2f} tok/s  (per-stream: {per_stream})", flush=True)

    mean_agg = sum(aggregates) / len(aggregates)
    print(f"\nMean aggregate throughput ({NUM_STREAMS} streams): {mean_agg:.2f} tok/s", flush=True)
    # Sanity: same prompt and model → bit-exact per-stream output should match
    if NUM_STREAMS >= 2:
        match = results[0]["gens"][:10] == results[1]["gens"][:10]
        print(f"streams produce identical first 10 tokens: {match}", flush=True)

    sock.close()


if __name__ == "__main__":
    main()
