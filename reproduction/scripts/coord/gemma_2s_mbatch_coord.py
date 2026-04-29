"""Gemma 4 E2B 2-stage 2-stream micro-batch coordinator.

Two threads, each running gemma_2s_coord-style logic on its own stream_id.
Single shared TCP socket (mutex'd around send+recv pair) — same approach as
mini_coord_mbatch.py for Llama. Per-stream stage_0 compile on alpha to work
around OV reset_state bug across multiple InferRequests sharing a
CompiledModel.

Verifies the paper's 1.66x Gemma 2-stream micro-batch claim against the v2
(rotary fix) and v2_beam (rotary fix + beam_idx Gather injection) shards.
"""
import os, socket, struct, time, threading, statistics
import numpy as np
import openvino as ov

STAGE0_DIR = os.environ.get("STAGE0_DIR", r"C:\cascadia\shards_e2b_cached_2s_v2\stage_0")
TOK_DIR = os.environ.get("TOK_DIR", r"C:\cascadia\shards_e2b_cached_2s_v2\tokenizer")
STAGE1_HOST = os.environ.get("STAGE1_HOST", "192.168.86.28")
STAGE1_PORT = int(os.environ.get("STAGE1_PORT", "19101"))
DEVICE = os.environ.get("DEVICE", "GPU")
GEMMA_PROMPT = os.environ.get("GEMMA_PROMPT", "What is the capital of France?")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "50"))
N_RUNS = int(os.environ.get("N_RUNS", "3"))
N_WARMUP = int(os.environ.get("N_WARMUP", "2"))
NUM_STREAMS = int(os.environ.get("NUM_STREAMS", "2"))


def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(65536, n - len(buf)))
        if not chunk:
            raise ConnectionError(f"peer closed at {len(buf)}/{n}")
        buf.extend(chunk)
    return bytes(buf)


class Pipeline:
    def __init__(self, sock):
        self.sock = sock
        self.lock = threading.Lock()

    def reset(self, stream_id):
        with self.lock:
            self.sock.sendall(struct.pack("<II", stream_id, 0))
            recv_exact(self.sock, 4)

    def forward(self, stream_id, hs, pos, ext_kv_pairs):
        msg = struct.pack("<II", stream_id, 1)
        msg += struct.pack("<I", pos.size) + pos.astype(np.int64).tobytes()
        msg += struct.pack("<II", hs.shape[1], hs.shape[2]) + np.ascontiguousarray(hs, dtype=np.float32).tobytes()
        for (k, v) in ext_kv_pairs:
            kv_seq = k.shape[2]; kv_hd = k.shape[3]
            msg += struct.pack("<II", kv_seq, kv_hd)
            msg += np.ascontiguousarray(k, dtype=np.float32).tobytes()
            msg += np.ascontiguousarray(v, dtype=np.float32).tobytes()
        with self.lock:
            self.sock.sendall(msg)
            vocab_size, seq_len = struct.unpack("<II", recv_exact(self.sock, 8))
            data = recv_exact(self.sock, 1 * seq_len * vocab_size * 4)
        return np.frombuffer(data, dtype=np.float32).reshape(1, seq_len, vocab_size)


def stream_decode(stream_id, stage0_req, has_beam, s0_compiled_for_this_stream, pipeline, input_ids, max_tokens, results, start_barrier):
    s0_outs = list(s0_compiled_for_this_stream.outputs)
    def split(res0):
        outs_by_port = {o: res0[o] for o in s0_outs}
        def find(name):
            for o in s0_outs:
                if any(name in n for n in o.get_names()):
                    return outs_by_port[o]
            raise KeyError(name)
        hs = find("hidden_states")
        kv = [(find("cross_kv.0.key"), find("cross_kv.0.value")),
              (find("cross_kv.1.key"), find("cross_kv.1.value"))]
        return hs, kv

    def init_state():
        for sv in stage0_req.query_state():
            shape = list(sv.state.shape)
            shape[0] = 1; shape[2] = 0
            sv.state = ov.Tensor(np.zeros(shape, dtype=np.float32))

    stage0_req.reset_state(); init_state()
    pipeline.reset(stream_id)
    start_barrier.wait()
    t0 = time.perf_counter()

    pos = np.arange(input_ids.shape[1], dtype=np.int64).reshape(1, -1)
    feed0 = {"input_ids": input_ids, "position_ids": pos}
    if has_beam: feed0["beam_idx"] = np.zeros(1, dtype=np.int32)
    res0 = stage0_req.infer(feed0)
    hs, kv = split(res0)
    logits = pipeline.forward(stream_id, hs, pos, kv)
    nt = int(np.argmax(logits[0, -1, :]))
    gens = [nt]
    cur_pos = input_ids.shape[1]
    for _ in range(max_tokens - 1):
        ids1 = np.array([[nt]], dtype=np.int64)
        pos1 = np.array([[cur_pos]], dtype=np.int64)
        feed0 = {"input_ids": ids1, "position_ids": pos1}
        if has_beam: feed0["beam_idx"] = np.zeros(1, dtype=np.int32)
        res0 = stage0_req.infer(feed0)
        hs, kv = split(res0)
        logits = pipeline.forward(stream_id, hs, pos1, kv)
        nt = int(np.argmax(logits[0, -1, :]))
        gens.append(nt); cur_pos += 1
    elapsed = time.perf_counter() - t0
    results[stream_id] = {"gens": gens, "elapsed": elapsed, "tok_s": max_tokens / elapsed}


def main():
    from transformers import AutoTokenizer
    print(f"OV {ov.__version__}", flush=True)
    print(f"NUM_STREAMS={NUM_STREAMS}  STAGE0_DIR={STAGE0_DIR}", flush=True)

    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    msgs = [{"role": "user", "content": GEMMA_PROMPT}]
    chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok.encode(chat, add_special_tokens=False)
    input_ids = np.array([ids], dtype=np.int64)
    print(f"prompt: {len(ids)} tokens", flush=True)

    core = ov.Core()
    print(f"Reading + compiling {NUM_STREAMS} stage_0 copies on {DEVICE}...", flush=True)
    s0_model = core.read_model(os.path.join(STAGE0_DIR, "openvino_model.xml"))
    s0_compiled = []
    s0_reqs = []
    t0 = time.perf_counter()
    for s in range(NUM_STREAMS):
        c = core.compile_model(s0_model, DEVICE)
        s0_compiled.append(c)
        s0_reqs.append(c.create_infer_request())
    print(f"  compile: {time.perf_counter()-t0:.1f}s", flush=True)

    s0_outs = list(s0_compiled[0].outputs)
    s0_in_names = [list(i.get_names())[0] for i in s0_compiled[0].inputs]
    has_beam = "beam_idx" in s0_in_names
    print(f"stage_0 inputs: {s0_in_names}  has_beam_idx={has_beam}", flush=True)

    print(f"Connecting to stage 1 worker @ {STAGE1_HOST}:{STAGE1_PORT}...", flush=True)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(300.0)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    sock.connect((STAGE1_HOST, STAGE1_PORT))
    pipe = Pipeline(sock)
    print("Connected.", flush=True)

    def run_batch():
        results = [None] * NUM_STREAMS
        barrier = threading.Barrier(NUM_STREAMS)
        threads = [threading.Thread(target=stream_decode,
                                    args=(s, s0_reqs[s], has_beam, s0_compiled[s], pipe, input_ids, MAX_TOKENS, results, barrier))
                   for s in range(NUM_STREAMS)]
        t0 = time.perf_counter()
        for t in threads: t.start()
        for t in threads: t.join()
        return results, time.perf_counter() - t0

    print(f"Warmup x{N_WARMUP}...", flush=True)
    for _ in range(N_WARMUP):
        run_batch()

    print(f"Timed x{N_RUNS} ({NUM_STREAMS} streams x {MAX_TOKENS} tokens):", flush=True)
    aggs = []
    for i in range(N_RUNS):
        results, total = run_batch()
        total_tokens = NUM_STREAMS * MAX_TOKENS
        agg = total_tokens / total
        per = [r["tok_s"] for r in results]
        aggs.append(agg)
        print(f"  run {i+1}: total {total_tokens} tok / {total:.3f}s = {agg:.2f} tok/s  per-stream={per}", flush=True)

    mean_agg = statistics.mean(aggs)
    sd = statistics.stdev(aggs) if len(aggs) > 1 else 0.0
    print(f"\nMean aggregate ({NUM_STREAMS} streams): {mean_agg:.2f} tok/s  sd={sd:.2f}", flush=True)
    if NUM_STREAMS >= 2:
        match = results[0]["gens"][:10] == results[1]["gens"][:10]
        print(f"streams produce identical first 10 tokens: {match}", flush=True)
    decoded = tok.decode(results[0]["gens"], skip_special_tokens=True)
    print(f"output: {decoded!r}", flush=True)


if __name__ == "__main__":
    main()
