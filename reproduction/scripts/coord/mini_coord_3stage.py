"""3-stage 2-node distributed decode — baseline (no spec, single stream).

alpha: stage_0 (embed + layers 0-10) local
charlie: stage_1 (layers 11-21, middle) remote via TCP
beta: stage_2 (layers 22-31 + head) remote via TCP

Protocol: coordinator (alpha) opens TCP connections to both charlie and beta.
Per forward:
  1. Run stage_0 on alpha → hidden_states
  2. Send hidden to charlie → recv hidden_middle
  3. Send hidden_middle to beta → recv logits
  4. argmax(logits) → next token

For now: single-stream baseline. Mirrors scripts/mini_coord_stage0.py's
structure but adds the extra middle hop.
"""
import os, socket, struct, sys, time, numpy as np, openvino as ov
from transformers import AutoTokenizer

TARGET_MODEL = r"C:\cascadia\models\llama-3.1-8b-int4"
STAGE0_SHARD = os.environ.get("STAGE0_SHARD", r"C:\cascadia\shards_3stage_v5_beam\stage_0")
STAGE1_HOST = os.environ.get("STAGE1_HOST", "192.168.86.28")   # charlie
STAGE1_PORT = int(os.environ.get("STAGE1_PORT", "19100"))
STAGE2_HOST = os.environ.get("STAGE2_HOST", "192.168.86.36")   # beta (refreshed 2026-04-26)
STAGE2_PORT = int(os.environ.get("STAGE2_PORT", "19100"))
PROMPT = os.environ.get("SPEC_PROMPT", "What is the capital of France?")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "128"))


def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(65536, n - len(buf)))
        if not chunk:
            raise ConnectionError(f"peer closed, got {len(buf)} of {n} bytes")
        buf.extend(chunk)
    return bytes(buf)


class RemoteStage:
    """Transport to a remote worker (stage 1 or stage 2). Tracks per-stream state.

    Same mini_worker_stage1.py protocol:
      Request: stream_id, op, logical_pos, attn_mask_len, attn_mask[int64],
               input_seq_len, hidden_size, hidden[float32]
      Response: seq_len, output_dim, output[float32]
    """

    def __init__(self, host, port, name=""):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(300.0)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        self.sock.connect((host, port))
        self.name = name

    def reset(self, stream_id=0):
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
        self.sock.sendall(msg)
        seq_len, output_dim = struct.unpack("<II", recv_exact(self.sock, 8))
        bytes_needed = seq_len * output_dim * 4
        data = recv_exact(self.sock, bytes_needed)
        return np.frombuffer(data, dtype=np.float32).reshape(1, seq_len, output_dim)

    def close(self):
        self.sock.close()


def main():
    print(f"STAGE1={STAGE1_HOST}:{STAGE1_PORT}  STAGE2={STAGE2_HOST}:{STAGE2_PORT}", flush=True)
    tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    input_ids = tok.encode(PROMPT, return_tensors="np").astype(np.int64)

    print(f"Loading stage_0 shard: {STAGE0_SHARD}", flush=True)
    core = ov.Core()
    stage0 = core.compile_model(core.read_model(os.path.join(STAGE0_SHARD, "openvino_model.xml")), "GPU")
    stage0_req = stage0.create_infer_request()
    print("Stage 0 ready.", flush=True)

    print(f"Connecting to stage 1 (charlie)...", flush=True)
    stage1 = RemoteStage(STAGE1_HOST, STAGE1_PORT, "stage_1")
    print(f"Connecting to stage 2 (beta)...", flush=True)
    stage2 = RemoteStage(STAGE2_HOST, STAGE2_PORT, "stage_2")

    # State tracker (same attn_mask/position_ids used on all three stages)
    valid_mask = np.ones(4096, dtype=np.int64)
    cache_len = 0
    logical_pos = 0

    def reset_all():
        nonlocal cache_len, logical_pos
        stage0_req.reset_state()
        stage1.reset(0)
        stage2.reset(0)
        valid_mask[:] = 1
        cache_len = 0
        logical_pos = 0

    def build_mask(n):
        total = cache_len + n
        attn = np.empty((1, total), dtype=np.int64)
        attn[0, :cache_len] = valid_mask[:cache_len]
        attn[0, cache_len:] = 1
        pos = np.arange(logical_pos, logical_pos + n, dtype=np.int64).reshape(1, -1)
        return attn, pos

    def target_forward(input_ids_batch, n):
        nonlocal cache_len, logical_pos
        attn, pos = build_mask(n)

        # Stage 0 on alpha
        stage0_req.infer({"input_ids": input_ids_batch, "attention_mask": attn,
                          "position_ids": pos, "beam_idx": np.zeros(1, dtype=np.int32)})
        hidden = stage0_req.get_output_tensor(0).data.copy()

        # Stage 1 on charlie (middle — outputs hidden)
        middle_hidden = stage1.forward(0, hidden, attn, pos)

        # Stage 2 on beta (final — outputs logits)
        logits = stage2.forward(0, middle_hidden, attn, pos)

        cache_len += n
        logical_pos += n
        return logits

    def decode(max_tokens):
        reset_all()
        n = input_ids.shape[1]
        logits = target_forward(input_ids, n)
        nt = int(np.argmax(logits[0, -1, :]))
        gens = [nt]
        for _ in range(1, max_tokens):
            logits = target_forward(np.array([[nt]], dtype=np.int64), 1)
            nt = int(np.argmax(logits[0, -1, :]))
            gens.append(nt)
        return gens

    print("Warmup (2 runs)...", flush=True)
    for _ in range(2):
        decode(MAX_TOKENS)

    print(f"Timed (3 runs of {MAX_TOKENS} tokens)...", flush=True)
    for i in range(3):
        t0 = time.perf_counter()
        gens = decode(MAX_TOKENS)
        dt = time.perf_counter() - t0
        print(f"  run {i+1}: {MAX_TOKENS}/{dt:.3f}s = {MAX_TOKENS/dt:.2f} tok/s", flush=True)
    print(f"  first10: {gens[:10]}", flush=True)

    stage1.close()
    stage2.close()


if __name__ == "__main__":
    main()
