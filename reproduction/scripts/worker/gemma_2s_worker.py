"""Gemma 4 E2B 2-stage v2 stage_1 worker.

Receives from coord:
  uint32: op (0 = reset, 1 = forward)
  if op == 1:
    uint32: position_ids_len
    int64[position_ids_len]: position_ids (1, position_ids_len)
    uint32: hidden_seq_len
    uint32: hidden_dim_total  # = hidden_size + (downstream_pli_count) * pli_dim, but for stage_1, downstream_pli=0, so dim_total = hidden_size + n_external_pli*pli_dim
    float32[1*hidden_seq_len*hidden_dim_total]: hidden_states_with_pli
    For each external KV pair (n_external_kv in stage_config):
      uint32: kv_seq_len
      uint32: kv_head_dim
      float32[1, num_kv_heads, kv_seq_len, kv_head_dim]: ext_key
      float32[1, num_kv_heads, kv_seq_len, kv_head_dim]: ext_value

Sends back:
  uint32: vocab_size
  uint32: seq_len
  float32[1, seq_len, vocab_size]: logits
"""
import os, socket, struct, sys, time, json
import numpy as np
import openvino as ov

SHARD_DIR = os.environ.get("SHARD_DIR", r"C:\cascadia\shards_e2b_cached_2s_v2_stage_1")
LISTEN_PORT = int(os.environ.get("LISTEN_PORT", "19101"))
DEVICE = os.environ.get("DEVICE", "GPU")


def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(65536, n - len(buf)))
        if not chunk:
            raise ConnectionError(f"peer closed at {len(buf)}/{n}")
        buf.extend(chunk)
    return bytes(buf)


def main():
    print(f"Loading {SHARD_DIR}", flush=True)
    with open(os.path.join(SHARD_DIR, "stage_config.json")) as f:
        meta = json.load(f)
    print(f"  meta: stateful={meta['stateful']} n_external_kv={meta['n_external_kv']} num_kv={meta['num_kv_heads']}", flush=True)

    core = ov.Core()
    print(f"  compiling on {DEVICE}...", flush=True)
    t0 = time.perf_counter()
    s = core.compile_model(os.path.join(SHARD_DIR, "openvino_model.xml"), DEVICE)
    print(f"  compile: {time.perf_counter()-t0:.1f}s", flush=True)
    req = s.create_infer_request()

    # Print I/O contract
    in_names = [list(i.get_names())[0] for i in s.inputs]
    out_names = [list(o.get_names())[0] for o in s.outputs]
    print(f"  inputs: {in_names}", flush=True)
    print(f"  outputs: {out_names}", flush=True)
    n_ext = meta["n_external_kv"]
    num_kv_heads = meta["num_kv_heads"]
    ext_dims = meta["external_kv_dims"]

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", LISTEN_PORT))
    srv.listen(1)
    print(f"Listening on 0.0.0.0:{LISTEN_PORT}", flush=True)

    while True:
        try:
            conn, addr = srv.accept()
        except OSError:
            break
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        print(f"accepted from {addr}", flush=True)
        n_req = 0
        try:
            while True:
                op = struct.unpack("<I", recv_exact(conn, 4))[0]
                if op == 0:
                    req.reset_state()
                    conn.sendall(struct.pack("<I", 0))
                    n_req += 1
                    print(f"  req#{n_req} reset", flush=True)
                    continue
                pos_len = struct.unpack("<I", recv_exact(conn, 4))[0]
                pos = np.frombuffer(recv_exact(conn, pos_len * 8), dtype=np.int64).reshape(1, -1)
                hs_seq = struct.unpack("<I", recv_exact(conn, 4))[0]
                hs_dim = struct.unpack("<I", recv_exact(conn, 4))[0]
                hs = np.frombuffer(recv_exact(conn, hs_seq * hs_dim * 4), dtype=np.float32).reshape(1, hs_seq, hs_dim)
                feed = {"hidden_states": hs, "position_ids": pos}
                if os.environ.get("DEBUG_TENSORS", "0") == "1":
                    print(f"  RX hs sum={hs.sum():.4f} shape={hs.shape}", flush=True)
                for ki in range(n_ext):
                    kv_seq = struct.unpack("<I", recv_exact(conn, 4))[0]
                    kv_hd = struct.unpack("<I", recv_exact(conn, 4))[0]
                    k = np.frombuffer(recv_exact(conn, num_kv_heads * kv_seq * kv_hd * 4), dtype=np.float32).reshape(1, num_kv_heads, kv_seq, kv_hd)
                    v = np.frombuffer(recv_exact(conn, num_kv_heads * kv_seq * kv_hd * 4), dtype=np.float32).reshape(1, num_kv_heads, kv_seq, kv_hd)
                    feed[f"external_kv.{ki}.key"] = k
                    feed[f"external_kv.{ki}.value"] = v
                    if os.environ.get("DEBUG_TENSORS", "0") == "1":
                        print(f"  RX kv{ki} k.sum={k.sum():.4f} v.sum={v.sum():.4f} shape={k.shape}", flush=True)
                req.infer(feed)
                logits = req.get_output_tensor(0).data
                seq_len = logits.shape[1]
                vocab_size = logits.shape[2]
                conn.sendall(struct.pack("<II", vocab_size, seq_len))
                conn.sendall(np.ascontiguousarray(logits, dtype=np.float32).tobytes())
                n_req += 1
                if n_req % 20 == 0:
                    print(f"  req#{n_req} (forward) hs_seq={hs_seq} pos={int(pos[0,0])}", flush=True)
        except ConnectionError as e:
            print(f"conn closed: {e}", flush=True)
        except Exception as e:
            print(f"FAIL after {n_req} reqs: {type(e).__name__}: {e}", flush=True)
        finally:
            try: conn.close()
            except Exception: pass


if __name__ == "__main__":
    main()
