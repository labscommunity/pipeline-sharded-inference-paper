"""Generic N-stage spec+mbatch coordinator. Stage 0 runs locally on the
coordinator node (alpha-equivalent); stages 1..N-1 run as remote workers
that all use mini_worker_stage1.py with the right shard.

Worker host/port topology comes from STAGE_HOSTS env var (comma-separated
"host:port" entries, one per remote stage in stage order). Example:

    STAGE_HOSTS="100.77.178.45:19100,100.127.88.82:19100,100.75.226.6:19100,100.113.0.105:19100,100.88.34.75:19100,100.73.141.16:19100"

Auto-detects the SEND_TOPK=1 sentinel from the FINAL stage worker (vocab
dim returned as 0). Intermediate stages always send full hidden_states
(out_dim ~= hidden_size, e.g. 8192 for Llama 3.1 70B).

Identical wire protocol per stage as mini_worker_stage1.py:
  Request:  stream_id, op, [logical_pos, attn_len, attn_mask[i64],
            input_seq_len, hidden_size, hidden[f32]]
  Response: out_seq, out_dim, output[f32]
            (or out_dim=0 sentinel for top-1 token IDs from final stage)
"""
import os, socket, struct, sys, time, threading, numpy as np, openvino as ov
from transformers import AutoTokenizer

sys.path.insert(0, r"C:\cascadia")
from cascadia.pipeline.spec_decode import MaskedReq

TARGET_MODEL = os.environ.get("TARGET_MODEL", r"C:\cascadia\models\llama-3.1-70b")
STAGE0_SHARD = os.environ.get("STAGE0_SHARD", r"C:\cascadia\shards_70b_v5_beam\stage_0")
DRAFT_MODEL  = os.environ.get("DRAFT_MODEL",  r"C:\cascadia\models\llama-3.2-1b-int4")
# Comma-separated list of host:port for remote stages, in stage order.
# E.g. "h1:p1,h2:p2,h3:p3,h4:p4,h5:p5,h6:p6" for a 7-stage pipeline (1 local + 6 remote).
STAGE_HOSTS = os.environ.get("STAGE_HOSTS", "")
PROMPT = os.environ.get("SPEC_PROMPT", "What is the capital of France?")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "128"))
NUM_STREAMS = int(os.environ.get("NUM_STREAMS", "2"))
K = int(os.environ.get("K", "3"))
WARMUP_RUNS = int(os.environ.get("WARMUP_RUNS", "2"))
TIMED_RUNS = int(os.environ.get("TIMED_RUNS", "3"))


def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(65536, n - len(buf)))
        if not chunk:
            raise ConnectionError(f"peer closed, got {len(buf)} of {n} bytes")
        buf.extend(chunk)
    return bytes(buf)


class RemoteStage:
    """Single remote worker. Thread-safe via a per-socket lock."""

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
                # Final stage SEND_TOPK=1 sentinel — pre-argmaxed token IDs.
                ids_bytes = recv_exact(self.sock, out_seq * 8)
                return np.frombuffer(ids_bytes, dtype=np.int64).reshape(1, out_seq)
            data = recv_exact(self.sock, out_seq * out_dim * 4)
        return np.frombuffer(data, dtype=np.float32).reshape(1, out_seq, out_dim)

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass


class DistributedNStageTargetMaskedReq:
    """Target wrapper matching MaskedReq API. Runs stage_0 locally, then
    chains through N-1 remote stages. The final stage's output may be
    full logits (1, seq_len, vocab_size) or top-1 IDs (1, seq_len)."""

    def __init__(self, stage0_req, remote_stages, stream_id):
        self.stage0 = stage0_req
        self.remotes = remote_stages  # list, in stage order (1, 2, ..., N-1)
        self.stream_id = stream_id
        self.valid_mask = np.ones(4096, dtype=np.int64)
        self.cache_len = 0
        self.logical_pos = 0

    def reset(self):
        self.stage0.reset_state()
        for r in self.remotes:
            r.reset(self.stream_id)
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

        for remote in self.remotes:
            hidden = remote.forward(self.stream_id, hidden, attn, pos)

        self.cache_len += n
        self.logical_pos += n
        return hidden  # shape (1, seq_len, vocab) full or (1, seq_len) top-1

    def rewind(self, k):
        if k <= 0:
            return
        self.valid_mask[self.cache_len - k : self.cache_len] = 0
        self.logical_pos -= k


def stream_spec_decode(stream_id, stage0_req, draft_req, draft_has_beam,
                       remote_stages, prompt_ids, max_tokens, k, results, start_barrier):
    target = DistributedNStageTargetMaskedReq(stage0_req, remote_stages, stream_id)
    draft = MaskedReq(draft_req, draft_has_beam)

    target.reset()
    draft.reset()

    start_barrier.wait()
    t0 = time.perf_counter()

    t_logits = target.feed(prompt_ids)
    draft.feed(prompt_ids)
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
    n_stages = 1 + len(stage_hosts)  # stage_0 local + N-1 remote
    print(f"NUM_STREAMS={NUM_STREAMS}  K={K}  N_STAGES={n_stages}", flush=True)
    for i, (h, p) in enumerate(stage_hosts):
        print(f"  remote stage {i+1}: {h}:{p}", flush=True)

    tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    input_ids = tok.encode(PROMPT, return_tensors="np").astype(np.int64)

    print(f"Loading stage_0 shard: {STAGE0_SHARD}", flush=True)
    core = ov.Core()
    stage0_model = core.read_model(os.path.join(STAGE0_SHARD, "openvino_model.xml"))
    print(f"Compiling {NUM_STREAMS} independent stage 0 copies on local GPU...", flush=True)
    stage0_reqs = []
    for s in range(NUM_STREAMS):
        c = core.compile_model(stage0_model, "GPU")
        stage0_reqs.append(c.create_infer_request())
    print(f"Stage 0 ready.", flush=True)

    print(f"Loading draft: {DRAFT_MODEL}", flush=True)
    draft_model = core.read_model(os.path.join(DRAFT_MODEL, "openvino_model.xml"))
    print(f"Compiling {NUM_STREAMS} independent draft copies on local GPU...", flush=True)
    draft_reqs = []
    draft_beam = None
    for s in range(NUM_STREAMS):
        c = core.compile_model(draft_model, "GPU")
        if draft_beam is None:
            draft_beam = any("beam_idx" in i.get_names() for i in c.inputs)
        draft_reqs.append(c.create_infer_request())
    print(f"Draft ready (beam_idx={draft_beam}).", flush=True)

    remotes = []
    for i, (h, p) in enumerate(stage_hosts):
        print(f"Connecting to remote stage {i+1} at {h}:{p}...", flush=True)
        remotes.append(RemoteStage(h, p, f"stage_{i+1}"))

    def run_batch():
        results = [None] * NUM_STREAMS
        barrier = threading.Barrier(NUM_STREAMS)
        threads = [threading.Thread(
            target=stream_spec_decode,
            args=(s, stage0_reqs[s], draft_reqs[s], draft_beam,
                  remotes, input_ids, MAX_TOKENS, K, results, barrier))
            for s in range(NUM_STREAMS)]

        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return results, time.perf_counter() - t0

    print(f"Warmup x{WARMUP_RUNS}...", flush=True)
    for _ in range(WARMUP_RUNS):
        run_batch()

    print(f"Timed x{TIMED_RUNS} ({NUM_STREAMS} streams x {MAX_TOKENS} tokens each, K={K}, {n_stages}-stage):", flush=True)
    aggregates = []
    last_results = None
    for i in range(TIMED_RUNS):
        results, total_elapsed = run_batch()
        last_results = results
        total_tokens = NUM_STREAMS * MAX_TOKENS
        agg = total_tokens / total_elapsed
        aggregates.append(agg)
        per_stream = [f"{r['tok_s']:.2f}(ar={r['accept_rate']:.1%})" for r in results]
        print(f"  run {i+1}: {total_tokens} tok / {total_elapsed:.3f} s = {agg:.2f} tok/s  "
              f"per-stream=[{', '.join(per_stream)}]", flush=True)

    mean_agg = sum(aggregates) / len(aggregates)
    print(f"\nMean aggregate throughput: {mean_agg:.2f} tok/s  ({NUM_STREAMS} streams x K={K} spec decode, {n_stages}-stage)", flush=True)
    if NUM_STREAMS >= 2 and last_results is not None:
        ref = last_results[0]["gens"][:10]
        all_match = all(last_results[i]["gens"][:10] == ref for i in range(NUM_STREAMS))
        print(f"all {NUM_STREAMS} streams produce identical first 10 tokens: {all_match}", flush=True)
    if last_results is not None:
        print(f"first10 stream 0: {last_results[0]['gens'][:10]}", flush=True)

    for r in remotes:
        r.close()


if __name__ == "__main__":
    main()
