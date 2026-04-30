#!/usr/bin/env python3
"""TCP round-trip latency benchmark between two nodes.

Measures raw TCP RTT for various payload sizes to decompose the
35ms per-hop overhead observed in the distributed pipeline.

Usage:
    # On the server node:
    python tcp_latency_bench.py server --port 9200

    # On the client node:
    python tcp_latency_bench.py client --host <server_ip> --port 9200

    # Single command (runs server on remote, client locally):
    # (Used by the orchestrator — see run instructions below)
"""

import argparse
import json
import socket
import struct
import time
import sys


def run_server(port, bufsize):
    """Echo server: receives payload, sends back 4-byte ACK."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    if bufsize > 0:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, bufsize)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, bufsize)
    srv.bind(("0.0.0.0", port))
    srv.listen(1)
    print(f"Server listening on port {port} (bufsize={bufsize})")

    conn, addr = srv.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    if bufsize > 0:
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, bufsize)
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, bufsize)
    print(f"Client connected from {addr}")

    try:
        while True:
            # Read 4-byte length header
            header = b""
            while len(header) < 4:
                chunk = conn.recv(4 - len(header))
                if not chunk:
                    return
                header += chunk
            payload_len = struct.unpack(">I", header)[0]

            if payload_len == 0:
                # Shutdown signal
                break

            # Read payload
            received = 0
            while received < payload_len:
                chunk = conn.recv(min(65536, payload_len - received))
                if not chunk:
                    return
                received += len(chunk)

            # Send ACK (4 bytes)
            conn.sendall(struct.pack(">I", payload_len))
    finally:
        conn.close()
        srv.close()


def run_client(host, port, bufsize):
    """Benchmark client: sends payloads of various sizes, measures RTT."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    if bufsize > 0:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, bufsize)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, bufsize)

    print(f"Connecting to {host}:{port} (bufsize={bufsize})...")
    sock.connect((host, port))
    print("Connected.")

    # Actual SO_SNDBUF/SO_RCVBUF after setting
    actual_sndbuf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
    actual_rcvbuf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
    print(f"Actual SO_SNDBUF={actual_sndbuf}, SO_RCVBUF={actual_rcvbuf}")

    sizes = [1, 100, 1024, 4096, 16384, 65536, 262144]
    iterations = 200
    warmup = 20
    results = {}

    for size in sizes:
        payload = bytes(size)  # zeroed buffer
        rtts = []

        for i in range(warmup + iterations):
            header = struct.pack(">I", size)

            t0 = time.perf_counter()
            sock.sendall(header + payload)

            # Read 4-byte ACK
            ack = b""
            while len(ack) < 4:
                chunk = sock.recv(4 - len(ack))
                if not chunk:
                    raise ConnectionError("Server closed")
                ack += chunk

            rtt_ms = (time.perf_counter() - t0) * 1000

            if i >= warmup:
                rtts.append(rtt_ms)

        rtts.sort()
        n = len(rtts)
        results[size] = {
            "size_bytes": size,
            "mean_ms": sum(rtts) / n,
            "p50_ms": rtts[n // 2],
            "p95_ms": rtts[int(n * 0.95)],
            "p99_ms": rtts[int(n * 0.99)],
            "min_ms": rtts[0],
            "max_ms": rtts[-1],
        }
        print(f"  {size:>7d} B: mean={results[size]['mean_ms']:.2f}ms "
              f"p50={results[size]['p50_ms']:.2f}ms "
              f"p95={results[size]['p95_ms']:.2f}ms "
              f"min={results[size]['min_ms']:.2f}ms")

    # Shutdown signal
    sock.sendall(struct.pack(">I", 0))
    sock.close()

    # Print summary as JSON
    print(f"\n{json.dumps(results, indent=2)}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    sp = sub.add_parser("server")
    sp.add_argument("--port", type=int, default=9200)
    sp.add_argument("--bufsize", type=int, default=0,
                    help="SO_SNDBUF/SO_RCVBUF size (0=OS default)")

    cp = sub.add_parser("client")
    cp.add_argument("--host", required=True)
    cp.add_argument("--port", type=int, default=9200)
    cp.add_argument("--bufsize", type=int, default=0,
                    help="SO_SNDBUF/SO_RCVBUF size (0=OS default)")

    args = parser.parse_args()
    if args.mode == "server":
        run_server(args.port, args.bufsize)
    else:
        run_client(args.host, args.port, args.bufsize)
