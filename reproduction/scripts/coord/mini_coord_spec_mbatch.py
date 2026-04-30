"""Full stack: distributed 2-node 2-stage target + spec decode K=3 + 2-stream micro-batching.

Architecture (all on 2 nodes: alpha = coordinator + stage 0 + draft; charlie = stage 1):

  Per stream (of NUM_STREAMS=2):
    - DistributedTargetMaskedReq: stage 0 InferRequest on alpha
        + transport to charlie for stage 1
        + shared mask-based rewind state across both stages
    - MaskedReq for the 1B draft model (all local on alpha)
    - spec_decode_greedy() loop, as in single-node, unchanged

  Pipeline transport (shared across streams, but thread-safe via socket lock):
    - DistributedPipeline — pure transport; coordinator passes pre-built
      attention_mask + position_ids so state lives in the target wrapper
      (not duplicated inside the pipeline)

Per-stream compile_model() both on coordinator (alpha) for stage 0 + draft
and on worker (charlie) for stage 1 to work around Discovery #21 (OV 2026.1
multi-InferRequest reset-state bug).
"""
import os, socket, struct, sys, time, threading, numpy as np, openvino as ov
from transformers import AutoTokenizer

sys.path.insert(0, r"C:\cascadia")
from cascadia.pipeline.spec_decode import MaskedReq

TARGET_MODEL = r"C:\cascadia\models\llama-3.1-8b-int4"
STAGE0_SHARD = os.environ.get("STAGE0_SHARD", r"C:\cascadia\shards_2stage_v5_beam\stage_0")
DRAFT_MODEL  = os.environ.get("DRAFT_MODEL",  r"C:\cascadia\models\llama-3.2-1b-int4")
STAGE1_HOST = os.environ.get("STAGE1_HOST", "192.168.86.28")
STAGE1_PORT = int(os.environ.get("STAGE1_PORT", "19100"))
PROMPT = os.environ.get("SPEC_PROMPT", "What is the capital of France?")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "128"))
NUM_STREAMS = int(os.environ.get("NUM_STREAMS", "2"))
K = int(os.environ.get("K", "3"))


def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(65536, n - len(buf)))
        if not chunk:
            raise ConnectionError(f"peer closed, got {len(buf)} of {n} bytes")
        buf.extend(chunk)
    return bytes(buf)


class DistributedPipeline:
    """Pure transport to stage 1 worker. Thread-safe via socket lock.
    Stateless — the target wrapper owns the mask/position state."""

    def __init__(self, sock):
        self.sock = sock
        self.lock = threading.Lock()

    def reset(self, stream_id):
        with self.lock:
            self.sock.sendall(struct.pack("<II", stream_id, 0))
            _, _ = struct.unpack("<II", recv_exact(self.sock, 8))

    def forward_stage1(self, stream_id, hidden, attn_mask, position_ids):
        new_tokens = hidden.shape[1]
        hidden_size = hidden.shape[2]
        total = attn_mask.shape[1]
        logical_pos = int(position_ids[0, 0])
        hidden_bytes = np.ascontiguousarray(hidden, dtype=np.float32).tobytes()

        msg = (
            struct.pack("<IIII", stream_id, 1, logical_pos, total)
            + attn_mask.astype(np.int64).tobytes()
            + struct.pack("<II", new_tokens, hidden_size)
            + hidden_bytes
        )
        with self.lock:
            self.sock.sendall(msg)
            seq_len, vocab_size = struct.unpack("<II", recv_exact(self.sock, 8))
            if vocab_size == 0:
                # SEND_TOPK=1 mode: worker pre-argmaxed; we got token IDs only.
                ids_bytes = recv_exact(self.sock, seq_len * 8)
                return np.frombuffer(ids_bytes, dtype=np.int64).reshape(1, seq_len)
            logits_bytes = recv_exact(self.sock, seq_len * vocab_size * 4)
        return np.frombuffer(logits_bytes, dtype=np.float32).reshape(1, seq_len, vocab_size)


class DistributedTargetMaskedReq:
    """Target wrapper matching MaskedReq API. Runs stage 0 on alpha + stage 1 remote."""

    def __init__(self, stage0_req, pipeline, stream_id):
        self.stage0 = stage0_req
        self.pipeline = pipeline
        self.stream_id = stream_id
        self.valid_mask = np.ones(4096, dtype=np.int64)
        self.cache_len = 0
        self.logical_pos = 0

    def reset(self):
        self.stage0.reset_state()
        self.pipeline.reset(self.stream_id)
        self.valid_mask[:] = 1
        self.cache_len = 0
        self.logical_pos = 0

    def feed(self, input_ids):
        n = input_ids.shape[1]
        total = self.cache_len + n
        if total > len(self.valid_mask):
            new_size = max(total * 2, len(self.valid_mask) * 2)
            new_mask = np.ones(new_size, dtype=np.int64)
            new_mask[:len(self.valid_mask)] = self.valid_mask
            self.valid_mask = new_mask
        attn = np.empty((1, total), dtype=np.int64)
        attn[0, :self.cache_len] = self.valid_mask[:self.cache_len]
        attn[0, self.cache_len:] = 1
        pos = np.arange(self.logical_pos, self.logical_pos + n, dtype=np.int64).reshape(1, -1)

        self.stage0.infer({"input_ids": input_ids, "attention_mask": attn,
                           "position_ids": pos, "beam_idx": np.zeros(1, dtype=np.int32)})
        hidden = self.stage0.get_output_tensor(0).data.copy()

        logits = self.pipeline.forward_stage1(self.stream_id, hidden, attn, pos)

        self.cache_len += n
        self.logical_pos += n
        return logits

    def rewind(self, k):
        if k <= 0:
            return
        self.valid_mask[self.cache_len - k : self.cache_len] = 0
        self.logical_pos -= k


def stream_spec_decode(stream_id, stage0_req, draft_req, draft_has_beam, pipeline,
                       prompt_ids, max_tokens, k, results, start_barrier):
    target = DistributedTargetMaskedReq(stage0_req, pipeline, stream_id)
    draft = MaskedReq(draft_req, draft_has_beam)

    target.reset()
    draft.reset()

    start_barrier.wait()
    t0 = time.perf_counter()

    # Spec decode (duplicates spec_decode_greedy so we can count stats per stream)
    t_logits = target.feed(prompt_ids)
    draft.feed(prompt_ids)
    # If worker sent top-1 (shape [1, seq_len] int64), use directly; otherwise argmax.
    if t_logits.ndim == 3:
        first = int(np.argmax(t_logits[0, -1, :]))
    else:
        first = int(t_logits[0, -1])
    gens = [first]
    prev_correction = first

    d_logits = draft.feed(np.array([[first]], dtype=np.int64))
    d_last_logit = d_logits[0, -1, :].copy()

    total_accepted = 0
    total_drafts = 0
    n_steps = 0

    while len(gens) < max_tokens:
        n_steps += 1

        drafts = [int(np.argmax(d_last_logit))]
        for i in range(1, k):
            if len(gens) + len(drafts) >= max_tokens:
                break
            d_logits = draft.feed(np.array([[drafts[i - 1]]], dtype=np.int64))
            drafts.append(int(np.argmax(d_logits[0, -1, :])))
        d_advanced = len(drafts) - 1
        total_drafts += len(drafts)

        verify = np.array([[prev_correction] + drafts], dtype=np.int64)
        t_logits = target.feed(verify)
        # Top-1 mode returns shape (1, seq_len) int64 already; otherwise argmax full logits.
        t_greedy = t_logits[0] if t_logits.ndim == 2 else np.argmax(t_logits[0], axis=-1)

        accepted = 0
        for i in range(len(drafts)):
            if int(t_greedy[i]) == drafts[i]:
                accepted += 1
            else:
                break
        total_accepted += accepted

        if accepted < len(drafts):
            correction = int(t_greedy[accepted])
        else:
            correction = int(t_greedy[len(drafts)])

        for tk in drafts[:accepted] + [correction]:
            if len(gens) >= max_tokens:
                break
            gens.append(tk)

        target.rewind(len(drafts) - accepted)

        if accepted < len(drafts):
            draft.rewind(d_advanced - accepted)
            d_logits = draft.feed(np.array([[correction]], dtype=np.int64))
        else:
            d_logits = draft.feed(np.array([[drafts[-1]]], dtype=np.int64))
            d_logits = draft.feed(np.array([[correction]], dtype=np.int64))
        d_last_logit = d_logits[0, -1, :].copy()
        prev_correction = correction

    elapsed = time.perf_counter() - t0
    results[stream_id] = {
        "gens": gens,
        "elapsed": elapsed,
        "tok_s": max_tokens / elapsed,
        "n_steps": n_steps,
        "drafts": total_drafts,
        "accepted": total_accepted,
        "accept_rate": total_accepted / max(total_drafts, 1),
    }


def main():
    print(f"NUM_STREAMS={NUM_STREAMS}  K={K}  STAGE1={STAGE1_HOST}:{STAGE1_PORT}", flush=True)
    tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    input_ids = tok.encode(PROMPT, return_tensors="np").astype(np.int64)

    print(f"Loading stage_0 shard: {STAGE0_SHARD}", flush=True)
    core = ov.Core()
    stage0_model = core.read_model(os.path.join(STAGE0_SHARD, "openvino_model.xml"))
    print(f"Compiling {NUM_STREAMS} independent stage 0 copies on alpha GPU...", flush=True)
    stage0_reqs = []
    for s in range(NUM_STREAMS):
        c = core.compile_model(stage0_model, "GPU")
        stage0_reqs.append(c.create_infer_request())
    print(f"Stage 0 ready.", flush=True)

    print(f"Loading draft: {DRAFT_MODEL}", flush=True)
    draft_model = core.read_model(os.path.join(DRAFT_MODEL, "openvino_model.xml"))
    print(f"Compiling {NUM_STREAMS} independent draft copies on alpha GPU...", flush=True)
    draft_reqs = []
    draft_beam = None
    for s in range(NUM_STREAMS):
        c = core.compile_model(draft_model, "GPU")
        if draft_beam is None:
            draft_beam = any("beam_idx" in i.get_names() for i in c.inputs)
        draft_reqs.append(c.create_infer_request())
    print(f"Draft ready (beam_idx={draft_beam}).", flush=True)

    print(f"Connecting to stage 1 worker...", flush=True)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(300.0)
    sock.connect((STAGE1_HOST, STAGE1_PORT))
    pipeline = DistributedPipeline(sock)

    def run_batch():
        results = [None] * NUM_STREAMS
        barrier = threading.Barrier(NUM_STREAMS)
        threads = [threading.Thread(
            target=stream_spec_decode,
            args=(s, stage0_reqs[s], draft_reqs[s], draft_beam, pipeline,
                  input_ids, MAX_TOKENS, K, results, barrier))
            for s in range(NUM_STREAMS)]

        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return results, time.perf_counter() - t0

    print("Warmup x2...", flush=True)
    for _ in range(2):
        run_batch()

    print(f"Timed x3 ({NUM_STREAMS} streams × {MAX_TOKENS} tokens each, K={K}):", flush=True)
    aggregates = []
    for i in range(3):
        results, total_elapsed = run_batch()
        total_tokens = NUM_STREAMS * MAX_TOKENS
        agg = total_tokens / total_elapsed
        aggregates.append(agg)
        per_stream = [f"{r['tok_s']:.2f}(ar={r['accept_rate']:.1%})" for r in results]
        print(f"  run {i+1}: {total_tokens} tok / {total_elapsed:.3f} s = {agg:.2f} tok/s  per-stream=[{', '.join(per_stream)}]", flush=True)

    mean_agg = sum(aggregates) / len(aggregates)
    print(f"\nMean aggregate throughput: {mean_agg:.2f} tok/s  ({NUM_STREAMS} streams × K={K} spec decode)", flush=True)
    if NUM_STREAMS >= 2:
        match = results[0]["gens"][:10] == results[1]["gens"][:10]
        print(f"streams produce identical first 10 tokens: {match}", flush=True)
    print(f"first10 stream 0: {results[0]['gens'][:10]}", flush=True)

    sock.close()


if __name__ == "__main__":
    main()
