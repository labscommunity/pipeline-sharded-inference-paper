"""Pipeline Coordinator — runs on the coordinator node.

Manages the full token generation loop:
1. Tokenize the prompt
2. Send embeddings into the pipeline (to stage 0's layers, then forward)
3. Receive generated tokens from the last stage
4. Repeat for autoregressive generation
"""

import numpy as np
import time
from dataclasses import dataclass, field

from ..model.loader import ModelShard
from ..model.ov_shard import OVShard
from ..config import PipelineConfig, StageConfig
from .activation_relay import ActivationClient, TransferStats


@dataclass
class GenerationMetrics:
    """Metrics from a single generation (one prompt)."""
    prompt_tokens: int = 0
    generated_tokens: int = 0
    ttft_ms: float = 0.0                     # Time to first token
    total_time_ms: float = 0.0
    per_token_latencies_ms: list[float] = field(default_factory=list)
    total_network_ms: float = 0.0
    total_compute_ms: float = 0.0

    @property
    def tokens_per_sec(self) -> float:
        if self.total_time_ms <= 0:
            return 0
        return self.generated_tokens / (self.total_time_ms / 1000)

    @property
    def avg_token_latency_ms(self) -> float:
        if not self.per_token_latencies_ms:
            return 0
        return sum(self.per_token_latencies_ms) / len(self.per_token_latencies_ms)


class PipelineCoordinator:
    """Coordinates token generation across a distributed pipeline."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.shard = None            # Stage 0 model shard (runs on coordinator)
        self.downstream_client = None  # Sends activations to stage 1
        self.result_server = None      # Receives tokens back from last stage

    def initialize(self):
        """Load model shard and set up network connections."""
        stage0 = self.config.stages[0]

        # Load model for stage 0 (coordinator's local shard)
        if stage0.shard_dir or self.config.shard_base_dir:
            import os
            shard_dir = stage0.shard_dir or os.path.join(
                self.config.shard_base_dir, "stage_0"
            )
            self.shard = OVShard(
                shard_dir=shard_dir,
                device=stage0.device,
                is_first_stage=True,
                is_last_stage=(self.config.total_stages == 1),
            )
        else:
            self.shard = ModelShard(
                model_path=self.config.model_local_path,
                layer_start=stage0.layer_start,
                layer_end=stage0.layer_end,
                device=stage0.device,
                is_first_stage=True,
                is_last_stage=(self.config.total_stages == 1),
            )
        self.shard.load()

        if self.config.total_stages > 1:
            # Connect to stage 1
            stage1 = self.config.stages[1]
            self.downstream_client = ActivationClient(stage1.host, stage1.port)

    def connect(self):
        """Establish connections to the pipeline."""
        if self.downstream_client:
            self.downstream_client.connect()

    def generate(self, prompt: str, max_new_tokens: int = None) -> tuple[str, GenerationMetrics]:
        """Generate text for a prompt through the distributed pipeline.

        Args:
            prompt: Input text
            max_new_tokens: Max tokens to generate (default from config)

        Returns:
            Tuple of (generated_text, metrics)
        """
        max_tokens = max_new_tokens or self.config.max_new_tokens
        metrics = GenerationMetrics()

        # Tokenize
        tokenizer = self.shard.tokenizer
        input_ids = tokenizer.encode(prompt, return_tensors="np")
        metrics.prompt_tokens = input_ids.shape[1]

        # Generate tokens autoregressively
        generated_ids = []
        gen_start = time.perf_counter()

        # Current sequence (grows with each generated token)
        current_ids = input_ids.copy()

        for step in range(max_tokens):
            token_start = time.perf_counter()

            # Stage 0: Embedding + local layers
            compute_start = time.perf_counter()
            hidden_states = self.shard.embed(current_ids)
            hidden_states = self.shard.forward_layers(hidden_states)
            stage0_compute_ms = (time.perf_counter() - compute_start) * 1000
            metrics.total_compute_ms += stage0_compute_ms

            if self.config.total_stages == 1:
                # Single-node: run LM head locally
                logits = self.shard.lm_head(hidden_states)
                next_token = int(np.argmax(logits[0, -1, :]))
            else:
                # Send activation to stage 1
                send_stats = self.downstream_client.send(hidden_states)
                metrics.total_network_ms += send_stats.send_time_ms

                # Receive token back through downstream (relayed by intermediate stages)
                token_array, recv_stats = self.downstream_client.recv()
                metrics.total_network_ms += recv_stats.recv_time_ms
                next_token = int(token_array[0])

            # Record timing
            token_ms = (time.perf_counter() - token_start) * 1000
            metrics.per_token_latencies_ms.append(token_ms)

            if step == 0:
                metrics.ttft_ms = token_ms

            # Append token
            generated_ids.append(next_token)
            current_ids = np.append(current_ids, [[next_token]], axis=1)

            # Check for EOS
            if next_token == tokenizer.eos_token_id:
                break

        metrics.generated_tokens = len(generated_ids)
        metrics.total_time_ms = (time.perf_counter() - gen_start) * 1000

        # Decode
        output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return output_text, metrics

    def shutdown(self):
        if self.downstream_client:
            self.downstream_client.close()
