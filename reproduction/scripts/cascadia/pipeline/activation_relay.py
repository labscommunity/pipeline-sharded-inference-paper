"""TCP-based activation tensor relay between pipeline stages.

This is intentionally simple: raw TCP with a minimal header. The
autoresearch agent will test alternatives (QUIC, UDP, shared memory,
compression, etc.) by swapping this module.

Protocol:
  [4 bytes: payload_length (uint32 big-endian)]
  [4 bytes: dtype enum (0=float32, 1=float16, 2=int8)]
  [4 bytes: dim0 (uint32)]
  [4 bytes: dim1 (uint32)]
  [4 bytes: dim2 (uint32)]
  [payload_length bytes: raw tensor data]
"""

import socket
import struct
import numpy as np
import time
from dataclasses import dataclass

DTYPE_MAP = {
    0: np.float32,
    1: np.float16,
    2: np.int8,
    3: np.int32,
}
DTYPE_REVERSE = {v: k for k, v in DTYPE_MAP.items()}

HEADER_SIZE = 20  # 4 + 4 + 4 + 4 + 4 bytes
RECV_BUFFER = 65536


@dataclass
class TransferStats:
    """Timing stats for a single activation transfer."""
    send_time_ms: float = 0.0
    recv_time_ms: float = 0.0
    bytes_transferred: int = 0


def send_activation(
    sock: socket.socket,
    tensor: np.ndarray,
) -> TransferStats:
    """Send an activation tensor over a connected TCP socket.

    Args:
        sock: Connected TCP socket
        tensor: Numpy array to send (any shape up to 3D)

    Returns:
        TransferStats with timing information
    """
    stats = TransferStats()

    # Ensure tensor is contiguous and get raw bytes
    tensor = np.ascontiguousarray(tensor)
    data = tensor.tobytes()
    dtype_code = DTYPE_REVERSE.get(tensor.dtype.type, 0)

    # Pad shape to 3D
    shape = list(tensor.shape)
    while len(shape) < 3:
        shape.insert(0, 1)

    # Build header
    header = struct.pack(
        ">IIIII",
        len(data),
        dtype_code,
        shape[0],
        shape[1],
        shape[2],
    )

    start = time.perf_counter()
    sock.sendall(header + data)
    stats.send_time_ms = (time.perf_counter() - start) * 1000
    stats.bytes_transferred = len(header) + len(data)

    return stats


def recv_activation(
    sock: socket.socket,
) -> tuple[np.ndarray, TransferStats]:
    """Receive an activation tensor from a connected TCP socket.

    Args:
        sock: Connected TCP socket

    Returns:
        Tuple of (numpy array, TransferStats)
    """
    stats = TransferStats()
    start = time.perf_counter()

    # Read header
    header_data = _recv_exact(sock, HEADER_SIZE)
    payload_len, dtype_code, d0, d1, d2 = struct.unpack(">IIIII", header_data)

    # Read payload
    payload = _recv_exact(sock, payload_len)

    stats.recv_time_ms = (time.perf_counter() - start) * 1000
    stats.bytes_transferred = HEADER_SIZE + payload_len

    # Reconstruct tensor
    dtype = DTYPE_MAP.get(dtype_code, np.float32)
    tensor = np.frombuffer(payload, dtype=dtype).reshape(d0, d1, d2)

    return tensor, stats


def _recv_exact(sock: socket.socket, num_bytes: int) -> bytes:
    """Receive exactly num_bytes from socket."""
    chunks = []
    received = 0
    while received < num_bytes:
        chunk = sock.recv(min(RECV_BUFFER, num_bytes - received))
        if not chunk:
            raise ConnectionError("Socket closed during recv")
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


class ActivationServer:
    """TCP server that receives activations on a port.

    Used by non-first pipeline stages to receive activations from
    the previous stage.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 9100):
        self.host = host
        self.port = port
        self.server_sock = None
        self.client_sock = None

    def start(self):
        """Start listening for connections."""
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Disable Nagle's algorithm for low-latency transfers
        self.server_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.server_sock.bind((self.host, self.port))
        self.server_sock.listen(1)
        print(f"ActivationServer listening on {self.host}:{self.port}")

    def accept(self):
        """Accept a connection from the upstream stage."""
        self.client_sock, addr = self.server_sock.accept()
        self.client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"Accepted connection from {addr}")

    def recv(self) -> tuple[np.ndarray, TransferStats]:
        """Receive an activation from the upstream stage."""
        return recv_activation(self.client_sock)

    def send(self, tensor: np.ndarray) -> TransferStats:
        """Send activation/token back (used by last stage to return results)."""
        return send_activation(self.client_sock, tensor)

    def close(self):
        if self.client_sock:
            self.client_sock.close()
        if self.server_sock:
            self.server_sock.close()


class ActivationClient:
    """TCP client that sends activations to the next stage."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.sock = None

    def connect(self, timeout: float = 30.0):
        """Connect to the downstream stage's ActivationServer."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(timeout)

        # Retry connection (downstream stage may not be ready yet)
        import time as _time
        start = _time.time()
        while _time.time() - start < timeout:
            try:
                self.sock.connect((self.host, self.port))
                print(f"Connected to downstream stage at {self.host}:{self.port}")
                return
            except ConnectionRefusedError:
                _time.sleep(0.5)
        raise TimeoutError(f"Could not connect to {self.host}:{self.port} within {timeout}s")

    def send(self, tensor: np.ndarray) -> TransferStats:
        """Send an activation to the downstream stage."""
        return send_activation(self.sock, tensor)

    def recv(self) -> tuple[np.ndarray, TransferStats]:
        """Receive result back from downstream (for last-stage token return)."""
        return recv_activation(self.sock)

    def close(self):
        if self.sock:
            self.sock.close()
