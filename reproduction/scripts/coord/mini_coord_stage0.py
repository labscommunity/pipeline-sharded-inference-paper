"""Minimal coordinator for 2-node 2-stage v5_beam bench.

Runs stage 0 locally (alpha iGPU) and sends activations to stage 1 worker
(charlie iGPU) over TCP. Supports:
  - Single-stream baseline decode
  - 2-stream micro-batching (interleave request A and B)
  - Spec decode with K=3 via cascadia.pipeline.spec_decode
  - All combinations (spec decode + micro-batching)

Usage:
  set MODE=baseline|mbatch|spec|spec_mbatch
  set STAGE1_HOST=192.168.86.28
  python mini_coord_stage0.py

For this first version: implement MODE=baseline only (single-stream, no spec).
Later PRs can add the other modes once baseline is validated.
"""
import os, socket, struct, sys, time, numpy as np, openvino as ov
from transformers import AutoTokenizer

sys.path.insert(0, r"C:\cascadia")

TARGET_MODEL = r"C:\cascadia\models\llama-3.1-8b-int4"
STAGE0_SHARD = os.environ.get("STAGE0_SHARD", r"C:\cascadia\shards_2stage_v5_beam\stage_0")
STAGE1_HOST = os.environ.get("STAGE1_HOST", "192.168.86.28")
STAGE1_PORT = int(os.environ.get("STAGE1_PORT", "19100"))
MODE = os.environ.get("MODE", "baseline")
PROMPT = os.environ.get("SPEC_PROMPT", "What is the capital of France?")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "128"))


def sendall(sock, data):
    sock.sendall(data)


def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(65536, n - len(buf)))
        if not chunk:
            raise ConnectionError(f"peer closed, got {len(buf)} of {n} bytes")
        buf.extend(chunk)
    return bytes(buf)


class DistributedStage1:
    """Remote proxy for stage 1. Tracks per-stream attention_mask and logical_pos."""

    def __init__(self, sock, num_streams=1):
        self.sock = sock
        self.num_streams = num_streams
        # Per-stream state
        self.valid_masks = [np.ones(4096, dtype=np.int64) for _ in range(num_streams)]
        self.cache_lens = [0] * num_streams
        self.logical_positions = [0] * num_streams

    def reset(self, stream_id=0):
        self.sock.sendall(struct.pack("<II", stream_id, 0))  # op=0
        _seq_len, _vocab = struct.unpack("<II", recv_exact(self.sock, 8))
        self.valid_masks[stream_id][:] = 1
        self.cache_lens[stream_id] = 0
        self.logical_positions[stream_id] = 0

    def forward(self, stream_id, hidden_states, new_token_count):
        """Forward hidden_states through stage 1 for stream_id, return logits."""
        # Build attention_mask (valid_mask prefix + ones for new tokens)
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

        hidden_states = np.ascontiguousarray(hidden_states, dtype=np.float32)
        assert hidden_states.shape == (1, new_token_count, 4096), f"bad shape {hidden_states.shape}"

        logical_pos = self.logical_positions[stream_id]
        attn_mask_bytes = attn_mask.tobytes()

        # Send request
        self.sock.sendall(
            struct.pack("<IIII", stream_id, 1, logical_pos, total)  # op=1
            + attn_mask_bytes
            + struct.pack("<II", new_token_count, 4096)
            + hidden_states.tobytes()
        )

        # Read response
        seq_len, vocab_size = struct.unpack("<II", recv_exact(self.sock, 8))
        logits_bytes = recv_exact(self.sock, seq_len * vocab_size * 4)
        logits = np.frombuffer(logits_bytes, dtype=np.float32).reshape(1, seq_len, vocab_size)

        self.cache_lens[stream_id] = total
        self.logical_positions[stream_id] += new_token_count

        return logits

    def rewind(self, stream_id, k):
        if k <= 0:
            return
        mask = self.valid_masks[stream_id]
        cache_len = self.cache_lens[stream_id]
        mask[cache_len - k:cache_len] = 0
        self.logical_positions[stream_id] -= k


def main():
    print(f"MODE={MODE}  STAGE1={STAGE1_HOST}:{STAGE1_PORT}", flush=True)
    tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    input_ids = tok.encode(PROMPT, return_tensors="np").astype(np.int64)

    print(f"Loading stage_0 shard: {STAGE0_SHARD}", flush=True)
    core = ov.Core()
    stage0 = core.compile_model(core.read_model(os.path.join(STAGE0_SHARD, "openvino_model.xml")), "GPU")

    stage0_req = stage0.create_infer_request()

    print(f"Connecting to stage_1 worker at {STAGE1_HOST}:{STAGE1_PORT}...", flush=True)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(300.0)
    sock.connect((STAGE1_HOST, STAGE1_PORT))
    stage1 = DistributedStage1(sock, num_streams=1)

    def stage0_forward(input_ids_batch, attn_mask, pos):
        stage0_req.infer({"input_ids": input_ids_batch, "attention_mask": attn_mask,
                          "position_ids": pos, "beam_idx": np.zeros(1, dtype=np.int32)})
        return stage0_req.get_output_tensor(0).data  # hidden_states

    def stage0_attn_mask(past_mask, cache_len_plus_new):
        """Build attention_mask for stage 0 (it doesn't use masked-rewind)."""
        return np.ones((1, cache_len_plus_new), dtype=np.int64)

    # We track stage 0's logical position separately
    s0_pos = 0
    s0_valid_mask = np.ones(4096, dtype=np.int64)
    s0_cache_len = 0

    def s0_attn(new_tokens):
        nonlocal s0_cache_len
        total = s0_cache_len + new_tokens
        mask = s0_valid_mask
        if total > len(mask):
            new_size = max(total * 2, len(mask) * 2)
            nm = np.ones(new_size, dtype=np.int64)
            nm[:len(mask)] = mask
            mask_ = nm
        else:
            mask_ = mask
        attn = np.empty((1, total), dtype=np.int64)
        attn[0, :s0_cache_len] = mask_[:s0_cache_len]
        attn[0, s0_cache_len:] = 1
        return attn

    # Baseline decode
    def decode(max_tokens):
        nonlocal s0_pos, s0_cache_len
        # Reset everything
        stage0_req.reset_state()
        s0_pos = 0
        s0_cache_len = 0
        s0_valid_mask[:] = 1
        stage1.reset(0)

        # Prefill
        n = input_ids.shape[1]
        attn = s0_attn(n)
        pos = np.arange(s0_pos, s0_pos + n, dtype=np.int64).reshape(1, -1)
        hidden = stage0_forward(input_ids, attn, pos)
        s0_cache_len = n
        s0_pos = n
        logits = stage1.forward(0, hidden, n)
        nt = int(np.argmax(logits[0, -1, :]))
        gens = [nt]

        for _ in range(1, max_tokens):
            ids = np.array([[nt]], dtype=np.int64)
            attn = s0_attn(1)
            pos = np.array([[s0_pos]], dtype=np.int64)
            hidden = stage0_forward(ids, attn, pos)
            s0_cache_len += 1
            s0_pos += 1
            logits = stage1.forward(0, hidden, 1)
            nt = int(np.argmax(logits[0, -1, :]))
            gens.append(nt)
        return gens

    # Warmup
    print("Warmup (2 runs)...", flush=True)
    for _ in range(2):
        decode(MAX_TOKENS)

    # Timed
    print(f"Timed (3 runs of {MAX_TOKENS} tokens)...", flush=True)
    for i in range(3):
        t0 = time.perf_counter()
        gens = decode(MAX_TOKENS)
        dt = time.perf_counter() - t0
        print(f"  run {i+1}: {MAX_TOKENS}/{dt:.3f}s = {MAX_TOKENS/dt:.2f} tok/s", flush=True)
    print(f"  first10: {gens[:10]}", flush=True)

    # Clean shutdown
    sock.close()


if __name__ == "__main__":
    main()
