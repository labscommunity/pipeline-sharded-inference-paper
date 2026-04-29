"""TCP latency proxy — inject per-packet delay between a client and server.

Listens on LOCAL_PORT on all interfaces. Each incoming connection opens a
matching outbound connection to REMOTE_HOST:REMOTE_PORT. Bytes are copied
in both directions with a configurable one-way delay applied to every
read/send pair. Real TCP semantics (Nagle, slow-start, retransmit) are
preserved — unlike the worker's time.sleep() which only adds dead-time
within a single-threaded recv/send path.

Usage:
  LOCAL_PORT=29100 REMOTE_HOST=192.168.86.28 REMOTE_PORT=19100 LATENCY_MS=50 python tcp_latency_proxy.py
"""
import os, socket, threading, time

LOCAL_PORT = int(os.environ.get("LOCAL_PORT", "29100"))
REMOTE_HOST = os.environ.get("REMOTE_HOST", "192.168.86.28")
REMOTE_PORT = int(os.environ.get("REMOTE_PORT", "19100"))
LATENCY_MS = int(os.environ.get("LATENCY_MS", "0"))


def pipe_with_delay(src, dst, direction):
    """Copy bytes src → dst, adding LATENCY_MS one-way delay per chunk."""
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            if LATENCY_MS > 0:
                time.sleep(LATENCY_MS / 1000.0)
            dst.sendall(data)
    except (ConnectionError, OSError) as e:
        pass
    finally:
        try:
            dst.shutdown(socket.SHUT_WR)
        except Exception:
            pass


def handle_client(client_sock, client_addr):
    print(f"[proxy] client connected from {client_addr}", flush=True)
    try:
        remote_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        remote_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        remote_sock.connect((REMOTE_HOST, REMOTE_PORT))

        client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        t1 = threading.Thread(target=pipe_with_delay, args=(client_sock, remote_sock, "c→s"), daemon=True)
        t2 = threading.Thread(target=pipe_with_delay, args=(remote_sock, client_sock, "s→c"), daemon=True)
        t1.start(); t2.start()
        t1.join(); t2.join()
    finally:
        try: client_sock.close()
        except: pass
        try: remote_sock.close()
        except: pass
        print(f"[proxy] session with {client_addr} ended", flush=True)


def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", LOCAL_PORT))
    server.listen(4)
    print(f"[proxy] listening on :{LOCAL_PORT} → {REMOTE_HOST}:{REMOTE_PORT}  (LATENCY_MS={LATENCY_MS})", flush=True)

    while True:
        client_sock, addr = server.accept()
        threading.Thread(target=handle_client, args=(client_sock, addr), daemon=True).start()


if __name__ == "__main__":
    main()
