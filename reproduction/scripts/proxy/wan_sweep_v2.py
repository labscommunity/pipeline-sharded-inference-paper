"""Real-WAN sweep v2 — uses release-time delay queue for accurate latency.

The v1 proxy slept per-chunk before forwarding, which serializes large messages
(500KB logits over multiple recv() calls each got delayed independently). Real
WAN with TCP cwnd lets multiple segments be in flight, so cumulative cost for
a large message is closer to one RTT than N_chunks * RTT.

This v2 uses a release-time delay queue: each chunk gets `release_at = now + lat`
and is forwarded by a separate sender thread when its release time arrives.
Multiple chunks can wait concurrently, matching tc-netem behavior.
"""
import os, socket, struct, sys, threading, time
from collections import deque
import numpy as np
import openvino as ov
from transformers import AutoTokenizer

CHARLIE_HOST = "192.168.86.28"
BETA_HOST    = "192.168.86.32"
WORKER_PORT  = 19100
PROXY_PORT_S1 = 29100
PROXY_PORT_S2 = 29200

STAGE0_SHARD = r"C:\cascadia\shards_3stage_v5_beam\stage_0"
TOK_PATH = r"C:\cascadia\models\llama-3.1-8b-int4"
PROMPT = "What is the capital of France?"
MAX_TOKENS = 64
N_RUNS = 2

_proxy_stop = threading.Event()


def _delay_queue_pipe(src, dst, latency_ms):
    """Receive from src, queue with release_at, sender thread forwards on time.
    Multiple chunks can be in-flight concurrently — closer to tc-netem behavior."""
    q = deque()
    q_lock = threading.Lock()
    src_done = threading.Event()
    lat = latency_ms / 1000.0

    def receiver():
        try:
            while not _proxy_stop.is_set():
                data = src.recv(65536)
                if not data:
                    break
                with q_lock:
                    q.append((time.monotonic() + lat, data))
        except (ConnectionError, OSError):
            pass
        src_done.set()

    def sender():
        try:
            while not _proxy_stop.is_set():
                with q_lock:
                    if q:
                        rel, _ = q[0]
                    else:
                        rel = None
                if rel is None:
                    if src_done.is_set():
                        break
                    time.sleep(0.0005)
                    continue
                wait = rel - time.monotonic()
                if wait > 0:
                    time.sleep(min(wait, 0.005))
                    continue
                with q_lock:
                    _, data = q.popleft()
                try:
                    dst.sendall(data)
                except (ConnectionError, OSError):
                    return
        finally:
            try: dst.shutdown(socket.SHUT_WR)
            except Exception: pass

    tr = threading.Thread(target=receiver, daemon=True)
    ts = threading.Thread(target=sender, daemon=True)
    tr.start(); ts.start()
    tr.join(); ts.join()


def _proxy_server(listen_port, remote_host, remote_port, latency_ms, name):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", listen_port))
    srv.listen(2)
    srv.settimeout(1.0)
    print(f"[proxy-{name}] :{listen_port} -> {remote_host}:{remote_port} (lat={latency_ms}ms one-way, queued)", flush=True)
    while not _proxy_stop.is_set():
        try:
            cli, addr = srv.accept()
        except socket.timeout:
            continue
        except OSError:
            break
        cli.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            rem = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            rem.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            rem.connect((remote_host, remote_port))
        except Exception as e:
            print(f"[proxy-{name}] FAIL upstream connect: {e}", flush=True)
            cli.close()
            continue
        threading.Thread(target=_delay_queue_pipe, args=(cli, rem, latency_ms), daemon=True).start()
        threading.Thread(target=_delay_queue_pipe, args=(rem, cli, latency_ms), daemon=True).start()
    try: srv.close()
    except Exception: pass


def start_proxies(latency_ms):
    _proxy_stop.clear()
    threading.Thread(target=_proxy_server, args=(PROXY_PORT_S1, CHARLIE_HOST, WORKER_PORT, latency_ms, "S1"), daemon=True).start()
    threading.Thread(target=_proxy_server, args=(PROXY_PORT_S2, BETA_HOST,    WORKER_PORT, latency_ms, "S2"), daemon=True).start()
    time.sleep(0.5)


def stop_proxies():
    _proxy_stop.set()
    time.sleep(1.5)


def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(65536, n - len(buf)))
        if not chunk:
            raise ConnectionError(f"peer closed at {len(buf)}/{n}")
        buf.extend(chunk)
    return bytes(buf)


class RemoteStage:
    def __init__(self, host, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(300.0)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        self.sock.connect((host, port))

    def reset(self, stream_id=0):
        self.sock.sendall(struct.pack("<II", stream_id, 0))
        struct.unpack("<II", recv_exact(self.sock, 8))

    def forward(self, stream_id, hidden, attn_mask, position_ids):
        n_t = hidden.shape[1]; hd = hidden.shape[2]
        total = attn_mask.shape[1]; lp = int(position_ids[0, 0])
        msg = (struct.pack("<IIII", stream_id, 1, lp, total)
               + attn_mask.astype(np.int64).tobytes()
               + struct.pack("<II", n_t, hd)
               + np.ascontiguousarray(hidden, dtype=np.float32).tobytes())
        self.sock.sendall(msg)
        seq_len, output_dim = struct.unpack("<II", recv_exact(self.sock, 8))
        data = recv_exact(self.sock, seq_len * output_dim * 4)
        return np.frombuffer(data, dtype=np.float32).reshape(1, seq_len, output_dim)

    def close(self):
        try: self.sock.close()
        except Exception: pass


def run_one(latency_ms, stage0_req, input_ids):
    print(f"\n=== latency {latency_ms} ms/hop ===", flush=True)
    start_proxies(latency_ms)
    try:
        s1 = RemoteStage("127.0.0.1", PROXY_PORT_S1)
        s2 = RemoteStage("127.0.0.1", PROXY_PORT_S2)
        rates = []
        for r in range(N_RUNS):
            stage0_req.reset_state(); s1.reset(0); s2.reset(0)
            cache_len = 0; logical_pos = 0
            t0 = time.perf_counter()
            n = input_ids.shape[1]
            attn = np.ones((1, cache_len + n), dtype=np.int64)
            pos = np.arange(logical_pos, logical_pos + n, dtype=np.int64).reshape(1, -1)
            stage0_req.infer({"input_ids": input_ids, "attention_mask": attn,
                              "position_ids": pos, "beam_idx": np.zeros(1, dtype=np.int32)})
            hidden = stage0_req.get_output_tensor(0).data.copy()
            mh = s1.forward(0, hidden, attn, pos)
            logits = s2.forward(0, mh, attn, pos)
            cache_len += n; logical_pos += n
            nt = int(np.argmax(logits[0, -1, :]))
            for _ in range(MAX_TOKENS - 1):
                ids1 = np.array([[nt]], dtype=np.int64)
                attn = np.ones((1, cache_len + 1), dtype=np.int64)
                pos = np.array([[logical_pos]], dtype=np.int64)
                stage0_req.infer({"input_ids": ids1, "attention_mask": attn,
                                  "position_ids": pos, "beam_idx": np.zeros(1, dtype=np.int32)})
                hidden = stage0_req.get_output_tensor(0).data.copy()
                mh = s1.forward(0, hidden, attn, pos)
                logits = s2.forward(0, mh, attn, pos)
                cache_len += 1; logical_pos += 1
                nt = int(np.argmax(logits[0, -1, :]))
            dt = time.perf_counter() - t0
            rate = MAX_TOKENS / dt
            rates.append(rate)
            print(f"  run {r+1}: {MAX_TOKENS}/{dt:.2f}s = {rate:.2f} tok/s", flush=True)
        s1.close(); s2.close()
        return rates
    finally:
        stop_proxies()


def main():
    print(f"OV {ov.__version__}", flush=True)
    tok = AutoTokenizer.from_pretrained(TOK_PATH)
    input_ids = tok.encode(PROMPT, return_tensors="np").astype(np.int64)
    print(f"Loading stage_0 shard...", flush=True)
    core = ov.Core()
    s0 = core.compile_model(core.read_model(os.path.join(STAGE0_SHARD, "openvino_model.xml")), "GPU")
    s0_req = s0.create_infer_request()
    print("Stage 0 ready.", flush=True)

    results = {}
    for lat in (0, 10, 50, 100):
        results[lat] = run_one(lat, s0_req, input_ids)

    print("\n=== summary (release-time delay queue) ===", flush=True)
    print(f"{'latency':>10} | {'mean tok/s':>12} | {'min':>6} | {'max':>6}", flush=True)
    for lat, rs in results.items():
        m = sum(rs) / len(rs)
        print(f"{lat:>7}ms | {m:>12.2f} | {min(rs):>6.2f} | {max(rs):>6.2f}", flush=True)


if __name__ == "__main__":
    main()
