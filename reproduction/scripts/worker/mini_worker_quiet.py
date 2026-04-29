"""Production worker for 2-node or 3-node distributed pipeline.

Per-stream independent compile_model (DISCOVERIES #21 workaround),
TCP_NODELAY + SO_KEEPALIVE, outer reconnect loop so a mid-session
connection abort re-accepts without restarting the whole worker.

Quiet: no per-request logging (that was in mini_worker_traced.py
and may have interacted badly with SSH stdout buffering + Windows
network stack on sustained traffic).
"""
import os, socket, struct, sys, time, numpy as np, openvino as ov

SHARD_DIR = os.environ.get("STAGE1_SHARD", r"C:\cascadia\shards_2stage_v5_beam\stage_1")
LISTEN_PORT = int(os.environ.get("LISTEN_PORT", "19100"))
DEVICE = os.environ.get("DEVICE", "GPU")
NUM_STREAMS = int(os.environ.get("NUM_STREAMS", "1"))
LATENCY_MS = int(os.environ.get("LATENCY_MS", "0"))


def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(65536, n - len(buf)))
        if not chunk:
            raise ConnectionError(f"peer closed, got {len(buf)} of {n} bytes")
        buf.extend(chunk)
    return bytes(buf)


def serve_once(reqs):
    """Accept one client connection and serve requests until it closes."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", LISTEN_PORT))
    server.listen(1)
    print(f"Listening on 0.0.0.0:{LISTEN_PORT}", flush=True)

    conn, addr = server.accept()
    print(f"Accepted from {addr}", flush=True)
    conn.settimeout(300.0)
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

    n_reqs = 0
    try:
        while True:
            header = recv_exact(conn, 4)
            if LATENCY_MS > 0:
                time.sleep(LATENCY_MS / 1000.0)
            stream_id = struct.unpack("<I", header)[0]
            op = struct.unpack("<I", recv_exact(conn, 4))[0]

            if stream_id >= NUM_STREAMS:
                print(f"ERROR: stream_id {stream_id} >= NUM_STREAMS {NUM_STREAMS}", flush=True)
                break

            n_reqs += 1

            if op == 0:
                reqs[stream_id].reset_state()
                conn.sendall(struct.pack("<II", 0, 0))
                continue

            logical_pos = struct.unpack("<I", recv_exact(conn, 4))[0]
            attn_mask_len = struct.unpack("<I", recv_exact(conn, 4))[0]
            attn_mask_bytes = recv_exact(conn, attn_mask_len * 8)
            attn_mask = np.frombuffer(attn_mask_bytes, dtype=np.int64).reshape(1, -1)
            input_seq_len = struct.unpack("<I", recv_exact(conn, 4))[0]
            hidden_size = struct.unpack("<I", recv_exact(conn, 4))[0]
            hidden_bytes = recv_exact(conn, input_seq_len * hidden_size * 4)
            hidden = np.frombuffer(hidden_bytes, dtype=np.float32).reshape(1, input_seq_len, hidden_size)
            pos = np.arange(logical_pos, logical_pos + input_seq_len, dtype=np.int64).reshape(1, -1)
            beam = np.zeros(1, dtype=np.int32)

            reqs[stream_id].infer({"hidden_states": hidden, "attention_mask": attn_mask,
                                   "position_ids": pos, "beam_idx": beam})
            logits = reqs[stream_id].get_output_tensor(0).data
            seq_len = logits.shape[1]
            output_dim = logits.shape[2]

            if LATENCY_MS > 0:
                time.sleep(LATENCY_MS / 1000.0)

            conn.sendall(struct.pack("<II", seq_len, output_dim))
            conn.sendall(np.ascontiguousarray(logits, dtype=np.float32).tobytes())
    finally:
        try:
            conn.close()
        except Exception:
            pass
        server.close()
        print(f"Session ended after {n_reqs} requests.", flush=True)


def main():
    print(f"Loading stage shard: {SHARD_DIR}", flush=True)
    core = ov.Core()
    model = core.read_model(os.path.join(SHARD_DIR, "openvino_model.xml"))
    print(f"Compiling {NUM_STREAMS} independent copies on {DEVICE}...", flush=True)
    reqs = []
    for s in range(NUM_STREAMS):
        c = core.compile_model(model, DEVICE)
        reqs.append(c.create_infer_request())
    print(f"Ready — NUM_STREAMS={NUM_STREAMS} LATENCY_MS={LATENCY_MS}", flush=True)

    # Keep re-accepting connections across client sessions
    while True:
        try:
            serve_once(reqs)
        except Exception as e:
            print(f"[outer] session errored: {type(e).__name__}: {str(e)[:200]}", flush=True)
            time.sleep(1.0)
        # Reset all stream states between sessions to keep things clean
        for r in reqs:
            r.reset_state()


if __name__ == "__main__":
    main()
