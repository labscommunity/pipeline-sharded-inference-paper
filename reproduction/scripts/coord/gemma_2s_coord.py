"""Gemma 4 E2B 2-stage coord — runs stage_0 locally, sends to remote stage_1.

Mirrors the gemma_bench_v2_2s.py protocol but with TCP between stage_0 (alpha)
and stage_1 (charlie). Validates the paper's "8.12 tok/s 2-stage multi-node"
number on the rotary-fixed v2 shards.
"""
import os, socket, struct, time, statistics
import numpy as np
import openvino as ov

STAGE0_DIR = os.environ.get("STAGE0_DIR", r"C:\cascadia\shards_e2b_cached_2s_v2\stage_0")
TOK_DIR = os.environ.get("TOK_DIR", r"C:\cascadia\shards_e2b_cached_2s_v2\tokenizer")
STAGE1_HOST = os.environ.get("STAGE1_HOST", "192.168.86.28")
STAGE1_PORT = int(os.environ.get("STAGE1_PORT", "19101"))
DEVICE = os.environ.get("DEVICE", "GPU")
PROMPT = os.environ.get("GEMMA_PROMPT", "What is the capital of France?")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "50"))
N_RUNS = int(os.environ.get("N_RUNS", "5"))
N_WARMUP = int(os.environ.get("N_WARMUP", "2"))


def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(65536, n - len(buf)))
        if not chunk:
            raise ConnectionError(f"peer closed at {len(buf)}/{n}")
        buf.extend(chunk)
    return bytes(buf)


def main():
    from transformers import AutoTokenizer
    print(f"OV {ov.__version__}", flush=True)
    print(f"Stage 0: {STAGE0_DIR}\nStage 1: {STAGE1_HOST}:{STAGE1_PORT}", flush=True)

    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    msgs = [{"role": "user", "content": PROMPT}]
    chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok.encode(chat, add_special_tokens=False)
    print(f"prompt: {len(ids)} tokens", flush=True)

    core = ov.Core()
    print(f"Compiling stage_0 on {DEVICE}...", flush=True)
    t0 = time.perf_counter()
    s0 = core.compile_model(os.path.join(STAGE0_DIR, "openvino_model.xml"), DEVICE)
    print(f"  {time.perf_counter()-t0:.1f}s", flush=True)
    r0 = s0.create_infer_request()

    s0_outs = list(s0.outputs)
    s0_in_names = [list(i.get_names())[0] for i in s0.inputs]
    has_beam = "beam_idx" in s0_in_names
    print(f"stage_0 inputs: {s0_in_names}  has_beam_idx={has_beam}", flush=True)
    print(f"stage_0 outputs: {[list(o.get_names())[0] for o in s0_outs]}", flush=True)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    sock.settimeout(120.0)
    print(f"Connecting to stage 1...", flush=True)
    sock.connect((STAGE1_HOST, STAGE1_PORT))
    print("Connected.", flush=True)

    def remote_reset():
        sock.sendall(struct.pack("<I", 0))
        recv_exact(sock, 4)

    def _csum(arr):
        a = np.ascontiguousarray(arr, dtype=np.float32)
        return float(a.sum()), a.shape, a.dtype

    DEBUG = os.environ.get("DEBUG_TENSORS", "0") == "1"

    def remote_forward(hs, pos, ext_kv_pairs):
        msg = struct.pack("<I", 1)
        msg += struct.pack("<I", pos.size) + pos.astype(np.int64).tobytes()
        hs_c = np.ascontiguousarray(hs, dtype=np.float32)
        msg += struct.pack("<II", hs.shape[1], hs.shape[2]) + hs_c.tobytes()
        for (k, v) in ext_kv_pairs:
            kv_seq = k.shape[2]; kv_hd = k.shape[3]
            msg += struct.pack("<II", kv_seq, kv_hd)
            msg += np.ascontiguousarray(k, dtype=np.float32).tobytes()
            msg += np.ascontiguousarray(v, dtype=np.float32).tobytes()
        if DEBUG:
            print(f"  TX hs sum={hs_c.sum():.4f} shape={hs.shape}", flush=True)
            for i, (k, v) in enumerate(ext_kv_pairs):
                kc = np.ascontiguousarray(k, dtype=np.float32)
                vc = np.ascontiguousarray(v, dtype=np.float32)
                print(f"  TX kv{i} k.sum={kc.sum():.4f} v.sum={vc.sum():.4f} shape={k.shape}", flush=True)
        sock.sendall(msg)
        vocab_size, seq_len = struct.unpack("<II", recv_exact(sock, 8))
        data = recv_exact(sock, 1 * seq_len * vocab_size * 4)
        return np.frombuffer(data, dtype=np.float32).reshape(1, seq_len, vocab_size)

    def split_stage0(res0, pos):
        outs_by_port = {o: res0[o] for o in s0_outs}
        def find(name_match):
            for o in s0_outs:
                if any(name_match in n for n in o.get_names()):
                    return outs_by_port[o]
            raise KeyError(name_match)
        hs = find("hidden_states")
        kv_pairs = [
            (find("cross_kv.0.key"), find("cross_kv.0.value")),
            (find("cross_kv.1.key"), find("cross_kv.1.value")),
        ]
        return hs, kv_pairs

    def _init_kv_state():
        """Set every state to shape (1, num_kv_heads, 0, head_dim) — works around the
        OV 2026.1 CPU plugin shape-inference bug where reset_state() leaves the
        ReadValue at batch=0 and the subsequent Concat (ReadValue, RoPE) fails.
        Harmless on GPU; required on CPU."""
        for sv in r0.query_state():
            shape = list(sv.state.shape)
            shape[0] = 1
            shape[2] = 0
            sv.state = ov.Tensor(np.zeros(shape, dtype=np.float32))

    def decode_one(ids, max_tokens):
        r0.reset_state()
        _init_kv_state()
        remote_reset()
        ids_np = np.array([ids], dtype=np.int64)
        pos_np = np.arange(len(ids), dtype=np.int64).reshape(1, -1)
        feed0 = {"input_ids": ids_np, "position_ids": pos_np}
        if has_beam:
            feed0["beam_idx"] = np.zeros(1, dtype=np.int32)
        res0 = r0.infer(feed0)
        hs, kv = split_stage0(res0, pos_np)
        logits = remote_forward(hs, pos_np, kv)
        nt = int(np.argmax(logits[0, -1, :]))
        gens = [nt]
        for step in range(max_tokens - 1):
            ids_one = np.array([[nt]], dtype=np.int64)
            pos_one = np.array([[len(ids) + step]], dtype=np.int64)
            feed0 = {"input_ids": ids_one, "position_ids": pos_one}
            if has_beam:
                feed0["beam_idx"] = np.zeros(1, dtype=np.int32)
            res0 = r0.infer(feed0)
            hs, kv = split_stage0(res0, pos_one)
            logits = remote_forward(hs, pos_one, kv)
            nt = int(np.argmax(logits[0, -1, :]))
            gens.append(nt)
        return gens

    print(f"\nwarmup x{N_WARMUP}...", flush=True)
    for _ in range(N_WARMUP):
        decode_one(ids, MAX_TOKENS)

    print(f"timed x{N_RUNS} ({MAX_TOKENS} tokens each):", flush=True)
    rates = []
    for i in range(N_RUNS):
        t0 = time.perf_counter()
        gens = decode_one(ids, MAX_TOKENS)
        dt = time.perf_counter() - t0
        rates.append(MAX_TOKENS / dt)
        print(f"  run {i+1}: {MAX_TOKENS/dt:.2f} tok/s  ({dt*1000/MAX_TOKENS:.1f} ms/tok wall)", flush=True)

    print(f"\nmean: {statistics.mean(rates):.2f} tok/s  sd={statistics.stdev(rates):.2f}", flush=True)
    print(f"output: {tok.decode(gens, skip_special_tokens=True)!r}", flush=True)


if __name__ == "__main__":
    main()
