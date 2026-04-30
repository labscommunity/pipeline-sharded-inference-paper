"""Target-only N-stage coordinator. Same pipeline infrastructure as
mini_coord_nstage_spec_mbatch.py but skips the draft model and spec
decode entirely. Used for correctness-baseline comparison: the spec
decoded output of mini_coord_nstage_spec_mbatch.py *should* match
the target-only output of this script bit-exactly (same prompt, same
target shards, same tokenizer).

Reuses the wire protocol and stage_0+remote stage classes from the
spec coord. Single-stream by design (no mbatch needed for a baseline).

Env vars:
  TARGET_MODEL    HF tokenizer path (default Llama 3.1 70B)
  STAGE0_SHARD    local stage_0 OpenVINO IR
  STAGE_HOSTS     comma-separated host:port for remote stages 1..N-1
  SPEC_PROMPT     prompt text (matches the spec coord default)
  MAX_TOKENS      tokens to decode (default 128)
  WARMUP_RUNS     warmup decodes (default 1; target-only is slower)
  TIMED_RUNS      timed decodes for tok/s (default 1)
"""
import os, socket, struct, sys, time, threading, numpy as np, openvino as ov
from transformers import AutoTokenizer

sys.path.insert(0, r"C:\cascadia")

TARGET_MODEL = os.environ.get("TARGET_MODEL", r"C:\cascadia\models\llama-3.1-70b")
STAGE0_SHARD = os.environ.get("STAGE0_SHARD", r"C:\cascadia\shards_70b_v5_beam_4stage\stage_0")
STAGE_HOSTS = os.environ.get("STAGE_HOSTS", "")
PROMPT = os.environ.get("SPEC_PROMPT", "What is the capital of France?")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "128"))
WARMUP_RUNS = int(os.environ.get("WARMUP_RUNS", "1"))
TIMED_RUNS = int(os.environ.get("TIMED_RUNS", "1"))


def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(65536, n - len(buf)))
        if not chunk:
            raise ConnectionError(f"peer closed, got {len(buf)} of {n} bytes")
        buf.extend(chunk)
    return bytes(buf)


class RemoteStage:
    def __init__(self, host, port, name=""):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(600.0)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        self.sock.connect((host, port))
        self.lock = threading.Lock()
        self.name = name

    def reset(self, stream_id):
        with self.lock:
            self.sock.sendall(struct.pack("<II", stream_id, 0))
            _, _ = struct.unpack("<II", recv_exact(self.sock, 8))

    def forward(self, stream_id, hidden, attn_mask, position_ids):
        new_tokens = hidden.shape[1]
        hidden_size = hidden.shape[2]
        total = attn_mask.shape[1]
        logical_pos = int(position_ids[0, 0])
        msg = (
            struct.pack("<IIII", stream_id, 1, logical_pos, total)
            + attn_mask.astype(np.int64).tobytes()
            + struct.pack("<II", new_tokens, hidden_size)
            + np.ascontiguousarray(hidden, dtype=np.float32).tobytes()
        )
        with self.lock:
            self.sock.sendall(msg)
            out_seq, out_dim = struct.unpack("<II", recv_exact(self.sock, 8))
            if out_dim == 0:
                ids_bytes = recv_exact(self.sock, out_seq * 8)
                return np.frombuffer(ids_bytes, dtype=np.int64).reshape(1, out_seq)
            data = recv_exact(self.sock, out_seq * out_dim * 4)
        return np.frombuffer(data, dtype=np.float32).reshape(1, out_seq, out_dim)

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass


class TargetForwardNoSpec:
    """Pipeline forward; same wire shape as the spec coord but no rewind
    needed since target-only always advances (no rejection)."""

    def __init__(self, stage0_req, remote_stages, stream_id):
        self.stage0 = stage0_req
        self.remotes = remote_stages
        self.stream_id = stream_id
        self.cache_len = 0
        self.logical_pos = 0

    def reset(self):
        self.stage0.reset_state()
        for r in self.remotes:
            r.reset(self.stream_id)
        self.cache_len = 0
        self.logical_pos = 0

    def feed(self, input_ids):
        n = input_ids.shape[1]
        total = self.cache_len + n
        attn = np.ones((1, total), dtype=np.int64)
        pos = np.arange(self.logical_pos, self.logical_pos + n, dtype=np.int64).reshape(1, -1)

        self.stage0.infer({"input_ids": input_ids, "attention_mask": attn,
                           "position_ids": pos, "beam_idx": np.zeros(1, dtype=np.int32)})
        hidden = self.stage0.get_output_tensor(0).data.copy()

        for remote in self.remotes:
            hidden = remote.forward(self.stream_id, hidden, attn, pos)

        self.cache_len += n
        self.logical_pos += n
        return hidden


def target_only_decode(target, prompt_ids, max_tokens):
    target.reset()
    t0 = time.perf_counter()
    logits = target.feed(prompt_ids)
    if logits.ndim == 3:
        first = int(np.argmax(logits[0, -1, :]))
    else:
        first = int(logits[0, -1])
    gens = [first]
    while len(gens) < max_tokens:
        logits = target.feed(np.array([[gens[-1]]], dtype=np.int64))
        if logits.ndim == 3:
            nxt = int(np.argmax(logits[0, -1, :]))
        else:
            nxt = int(logits[0, -1])
        gens.append(nxt)
    return gens, time.perf_counter() - t0


def parse_stage_hosts(s):
    if not s:
        raise ValueError("STAGE_HOSTS env var required (comma-separated host:port list)")
    out = []
    for entry in s.split(","):
        entry = entry.strip()
        if not entry:
            continue
        host, _, port = entry.partition(":")
        out.append((host, int(port) if port else 19100))
    return out


def main():
    stage_hosts = parse_stage_hosts(STAGE_HOSTS)
    n_stages = 1 + len(stage_hosts)
    print(f"TARGET_ONLY  N_STAGES={n_stages}  MAX_TOKENS={MAX_TOKENS}", flush=True)
    for i, (h, p) in enumerate(stage_hosts):
        print(f"  remote stage {i+1}: {h}:{p}", flush=True)

    tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    input_ids = tok.encode(PROMPT, return_tensors="np").astype(np.int64)
    print(f"prompt tokens: {input_ids.shape[1]}", flush=True)

    print(f"Loading stage_0 shard: {STAGE0_SHARD}", flush=True)
    core = ov.Core()
    stage0_model = core.read_model(os.path.join(STAGE0_SHARD, "openvino_model.xml"))
    print(f"Compiling stage 0 on local GPU (single stream, target-only)...", flush=True)
    compiled0 = core.compile_model(stage0_model, "GPU")
    stage0_req = compiled0.create_infer_request()
    print(f"Stage 0 ready.", flush=True)

    remotes = []
    for i, (h, p) in enumerate(stage_hosts):
        print(f"Connecting to remote stage {i+1} at {h}:{p}...", flush=True)
        remotes.append(RemoteStage(h, p, f"stage_{i+1}"))

    target = TargetForwardNoSpec(stage0_req, remotes, stream_id=0)

    print(f"Warmup x{WARMUP_RUNS}...", flush=True)
    for _ in range(WARMUP_RUNS):
        target_only_decode(target, input_ids, max(8, min(16, MAX_TOKENS)))

    print(f"Timed x{TIMED_RUNS} ({MAX_TOKENS} tokens, target-only, {n_stages}-stage):", flush=True)
    last_gens = None
    for i in range(TIMED_RUNS):
        gens, elapsed = target_only_decode(target, input_ids, MAX_TOKENS)
        last_gens = gens
        print(f"  run {i+1}: {MAX_TOKENS} tok / {elapsed:.3f} s = {MAX_TOKENS/elapsed:.2f} tok/s", flush=True)

    if last_gens is not None:
        print(f"first10: {last_gens[:10]}", flush=True)
        print(f"full {len(last_gens)}: {last_gens}", flush=True)
        try:
            decoded = tok.decode(last_gens, skip_special_tokens=True)
            print(f"decoded: {decoded!r}", flush=True)
        except Exception as e:
            print(f"decode failed: {e}", flush=True)

    for r in remotes:
        r.close()


if __name__ == "__main__":
    main()
