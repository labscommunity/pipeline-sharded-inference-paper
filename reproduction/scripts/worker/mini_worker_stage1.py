"""Traced version of mini_worker_stage1.py — logs every request shape."""
import os, socket, struct, sys, time, numpy as np, openvino as ov


SHARD_DIR = os.environ.get("STAGE1_SHARD", r"C:\cascadia\shards_2stage_v5_beam\stage_1")
LISTEN_PORT = int(os.environ.get("LISTEN_PORT", "19100"))
DEVICE = os.environ.get("DEVICE", "GPU")
NUM_STREAMS = int(os.environ.get("NUM_STREAMS", "2"))
# Simulated one-way network latency to/from the coordinator (ms). Applied
# on request receive AND response send, so total RTT added ~ 2 x LATENCY_MS.
LATENCY_MS = int(os.environ.get("LATENCY_MS", "0"))
# 0 = send full logits (paper default). 1 = compute argmax server-side and
# send only top-1 token IDs (correctness-preserving for greedy spec decode,
# bandwidth drops from seq_len*vocab_size*4 bytes to seq_len*8 bytes ~
# 256000x compression at vocab 128k). Coord auto-detects via vocab_size=0
# in the response header.
SEND_TOPK = int(os.environ.get("SEND_TOPK", "0"))


def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(65536, n - len(buf)))
        if not chunk:
            raise ConnectionError(f"peer closed, got {len(buf)} of {n} bytes")
        buf.extend(chunk)
    return bytes(buf)


def main():
    print(f"Loading stage_1 shard: {SHARD_DIR}", flush=True)
    core = ov.Core()
    # WORKAROUND: separate compile_model per stream, since sharing a
    # CompiledModel across multiple InferRequest + reset_state()
    # seems to get into inconsistent internal state on this OV build.
    model = core.read_model(os.path.join(SHARD_DIR, "openvino_model.xml"))
    print(f"Compiling {NUM_STREAMS} independent copies on {DEVICE}...", flush=True)
    reqs = []
    for s in range(NUM_STREAMS):
        compiled = core.compile_model(model, DEVICE)
        reqs.append(compiled.create_infer_request())
    print(f"Created {NUM_STREAMS} independent InferRequests.", flush=True)

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

    req_counter = 0
    while True:
        try:
            header = recv_exact(conn, 4)
        except ConnectionError:
            print("Connection closed by peer.", flush=True)
            break
        # Simulate one-way network latency on receive (request arrived late)
        if LATENCY_MS > 0:
            time.sleep(LATENCY_MS / 1000.0)
        stream_id = struct.unpack("<I", header)[0]
        op = struct.unpack("<I", recv_exact(conn, 4))[0]

        if stream_id >= NUM_STREAMS:
            print(f"ERROR: stream_id {stream_id} >= NUM_STREAMS {NUM_STREAMS}", flush=True)
            break

        req_counter += 1

        if op == 0:
            reqs[stream_id].reset_state()
            conn.sendall(struct.pack("<II", 0, 0))
            print(f"req#{req_counter:04d} stream={stream_id} op=reset ACK", flush=True)
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

        print(f"req#{req_counter:04d} stream={stream_id} logical_pos={logical_pos} attn_len={attn_mask_len} input_len={input_seq_len} hidden_shape={hidden.shape} attn_shape={attn_mask.shape} pos_shape={pos.shape}", flush=True)

        try:
            reqs[stream_id].infer({"hidden_states": hidden, "attention_mask": attn_mask,
                                   "position_ids": pos, "beam_idx": beam})
            logits = reqs[stream_id].get_output_tensor(0).data
        except Exception as e:
            print(f"INFER ERROR stream={stream_id}: {type(e).__name__}: {str(e)[:300]}", flush=True)
            raise

        seq_len = logits.shape[1]
        # Output for non-final stages is hidden_states (vocab_size dim is
        # actually hidden_size); always pass through. For the final stage
        # output is logits and SEND_TOPK can compress it.
        out_dim = logits.shape[2]
        is_logits_like = (out_dim > 32000)  # heuristic: any stage outputting >32k features is logits

        if LATENCY_MS > 0:
            time.sleep(LATENCY_MS / 1000.0)

        if SEND_TOPK == 1 and is_logits_like:
            argmax_ids = np.argmax(logits[0], axis=-1).astype(np.int64)
            conn.sendall(struct.pack("<II", seq_len, 0))  # vocab_size=0 sentinel
            conn.sendall(argmax_ids.tobytes())
        else:
            conn.sendall(struct.pack("<II", seq_len, out_dim))
            conn.sendall(np.ascontiguousarray(logits, dtype=np.float32).tobytes())

    try:
        conn.close()
    except Exception:
        pass
    server.close()


if __name__ == "__main__":
    # If the connection drops mid-session, restart the accept loop so the
    # coordinator can reconnect without restarting the whole worker
    # (GPU recompile of stage_1/stage_2 takes ~60s, so we keep the server alive).
    while True:
        try:
            main()
        except Exception as e:
            print(f"[outer] main() exited with {type(e).__name__}: {e}", flush=True)
            print(f"[outer] restarting main() loop", flush=True)
            import time as _t
            _t.sleep(1.0)
