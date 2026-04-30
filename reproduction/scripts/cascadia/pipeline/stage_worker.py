"""Pipeline stage worker — runs on each node in the pipeline.

Each worker:
1. Loads its assigned model layers
2. Listens for activations from the upstream stage (or tokenized input if stage 0)
3. Runs forward pass through its layers
4. Sends output activations to the downstream stage (or returns token if last stage)
"""

import numpy as np
import time
import json
import argparse
from pathlib import Path

from ..model.loader import ModelShard
from ..model.cascadia_shard import CascadiaShard
from ..config import StageConfig
from .activation_relay import ActivationServer, ActivationClient, TransferStats


# OVShard is imported lazily so coordinator-only / cascadia-only hosts
# don't need OpenVINO installed. Tests can monkey-patch this attribute
# to bypass the lazy import.
OVShard = None


def _resolve_ov_shard_class():
    global OVShard
    if OVShard is not None:
        return OVShard
    from ..model.ov_shard import OVShard as _OVShard
    OVShard = _OVShard
    return OVShard


class StageWorker:
    """A single pipeline stage worker."""

    def __init__(self, config: StageConfig, total_stages: int, model_path: str,
                 shard_dir: str = None):
        self.config = config
        self.total_stages = total_stages
        self.stage_index = None  # Set externally based on pipeline ordering

        self.is_first_stage = False  # Will be set by coordinator
        self.is_last_stage = False

        # Model dispatch:
        #   device == "CASCADIA_GPU" → CascadiaShard (cascadia-megakernel
        #     kernel adapter; OpenCL when wired, torch fallback today)
        #   shard_dir set            → OVShard (compiled OpenVINO IR)
        #   else                     → ModelShard (PyTorch eager)
        self._shard_dir = shard_dir or config.shard_dir
        if config.device == "CASCADIA_GPU":
            self.shard = CascadiaShard(
                model_path=model_path,
                layer_start=config.layer_start,
                layer_end=config.layer_end,
                device=config.device,
                is_first_stage=self.is_first_stage,
                is_last_stage=self.is_last_stage,
            )
        elif self._shard_dir:
            ov_cls = _resolve_ov_shard_class()
            self.shard = ov_cls(
                shard_dir=self._shard_dir,
                device=config.device,
                is_first_stage=self.is_first_stage,
                is_last_stage=self.is_last_stage,
            )
        else:
            self.shard = ModelShard(
                model_path=model_path,
                layer_start=config.layer_start,
                layer_end=config.layer_end,
                device=config.device,
                is_first_stage=self.is_first_stage,
                is_last_stage=self.is_last_stage,
            )

        # Network
        self.server = None    # Receives from upstream
        self.client = None    # Sends to downstream

        # Metrics
        self.total_compute_ms = 0.0
        self.total_network_ms = 0.0
        self.tokens_processed = 0

    def initialize(
        self,
        stage_index: int,
        listen_port: int,
        downstream_host: str = None,
        downstream_port: int = None,
    ):
        """Initialize network connections.

        Args:
            stage_index: This stage's index in the pipeline (0-based)
            listen_port: Port to listen for upstream activations
            downstream_host: Host of the next stage (None if last stage)
            downstream_port: Port of the next stage (None if last stage)
        """
        self.stage_index = stage_index
        self.is_first_stage = (stage_index == 0)
        self.is_last_stage = (downstream_host is None)
        self.shard.is_first_stage = self.is_first_stage
        self.shard.is_last_stage = self.is_last_stage

        # Recreate OVShard with correct flags if needed (flags affect load behavior)
        if self._shard_dir:
            self.shard = OVShard(
                shard_dir=self._shard_dir,
                device=self.config.device,
                is_first_stage=self.is_first_stage,
                is_last_stage=self.is_last_stage,
            )

        # Load model
        self.shard.load()

        # Start activation server (all stages listen for input)
        self.server = ActivationServer(port=listen_port)
        self.server.start()

        # Connect to downstream (if not last stage)
        if not self.is_last_stage:
            self.client = ActivationClient(downstream_host, downstream_port)

    def accept_upstream(self):
        """Accept connection from coordinator or upstream stage."""
        self.server.accept()

    def connect_downstream(self):
        """Connect to the next pipeline stage."""
        if self.client:
            self.client.connect()

    def process_one_token(self) -> tuple[np.ndarray | int, dict]:
        """Process one forward pass: receive activation, compute, forward.

        Returns:
            If last stage: (token_id, timing_dict)
            If not last stage: (output_activation, timing_dict)
        """
        timings = {}

        # Receive input activation from upstream
        activation, recv_stats = self.server.recv()
        timings["recv_ms"] = recv_stats.recv_time_ms
        timings["recv_bytes"] = recv_stats.bytes_transferred

        # Forward pass through our layers
        compute_start = time.perf_counter()
        output = self.shard.forward_layers(activation)
        timings["compute_ms"] = (time.perf_counter() - compute_start) * 1000

        self.total_compute_ms += timings["compute_ms"]
        self.total_network_ms += timings["recv_ms"]
        self.tokens_processed += 1

        if self.is_last_stage:
            # Run LM head and sample token
            logits = self.shard.lm_head(output)
            # Greedy sampling: take argmax of last position
            token_id = int(np.argmax(logits[0, -1, :]))

            # Send token back upstream (relayed hop-by-hop to coordinator)
            token_array = np.array([token_id], dtype=np.int32)
            send_stats = self.server.send(token_array)
            timings["send_ms"] = send_stats.send_time_ms

            return token_id, timings
        else:
            # Forward activation to downstream stage
            send_stats = self.client.send(output)
            timings["send_ms"] = send_stats.send_time_ms
            timings["send_bytes"] = send_stats.bytes_transferred
            self.total_network_ms += timings["send_ms"]

            # Relay token from downstream back upstream
            token_array, relay_recv = self.client.recv()
            relay_send = self.server.send(token_array)
            timings["relay_ms"] = relay_recv.recv_time_ms + relay_send.send_time_ms
            self.total_network_ms += timings["relay_ms"]

            return output, timings

    def get_metrics(self) -> dict:
        """Return accumulated metrics."""
        total_time = self.total_compute_ms + self.total_network_ms
        return {
            "stage_index": self.stage_index,
            "node_id": self.config.node_id,
            "layers": f"{self.config.layer_start}-{self.config.layer_end}",
            "tokens_processed": self.tokens_processed,
            "total_compute_ms": self.total_compute_ms,
            "total_network_ms": self.total_network_ms,
            "compute_pct": (self.total_compute_ms / total_time * 100) if total_time > 0 else 0,
            "avg_compute_ms_per_token": (self.total_compute_ms / self.tokens_processed) if self.tokens_processed > 0 else 0,
        }

    def shutdown(self):
        if self.server:
            self.server.close()
        if self.client:
            self.client.close()


def main():
    """Entry point for running a stage worker as a standalone process."""
    parser = argparse.ArgumentParser(description="CascadiaInfer Stage Worker")
    parser.add_argument("--config", required=True, help="Path to node config JSON")
    args = parser.parse_args()

    with open(args.config) as f:
        data = json.load(f)

    stage_config = StageConfig(
        node_id=data["node_id"],
        host=data.get("host", "0.0.0.0"),
        port=data["listen_port"],
        layer_start=data["layer_start"],
        layer_end=data["layer_end"],
        device=data.get("device", "CPU"),
    )

    worker = StageWorker(
        config=stage_config,
        total_stages=data["total_stages"],
        model_path=data.get("model_path", ""),
        shard_dir=data.get("shard_dir"),
    )

    worker.initialize(
        stage_index=data["stage_index"],
        listen_port=data["listen_port"],
        downstream_host=data.get("downstream_host"),
        downstream_port=data.get("downstream_port"),
    )

    print(f"Stage {data['stage_index']} ready. Waiting for upstream connection...")
    worker.accept_upstream()

    if not worker.is_last_stage:
        worker.connect_downstream()

    print(f"Stage {data['stage_index']} running. Processing tokens...")

    # Main loop: process tokens until connection closes
    try:
        while True:
            worker.process_one_token()
    except (ConnectionError, BrokenPipeError):
        print("Connection closed. Shutting down.")
    finally:
        metrics = worker.get_metrics()
        print(f"\nStage metrics: {json.dumps(metrics, indent=2)}")
        worker.shutdown()


if __name__ == "__main__":
    main()
