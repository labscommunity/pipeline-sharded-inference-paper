"""Configuration for the MVP scaffold."""

from dataclasses import dataclass, field
from typing import Optional
import yaml
import json


@dataclass
class StageConfig:
    """Config for a single pipeline stage."""
    node_id: str
    host: str
    port: int                  # TCP port this stage listens on
    layer_start: int
    layer_end: int             # Exclusive: layers [start, end)
    device: str = "CPU"        # OpenVINO device: "CPU", "GPU", "NPU"
    shard_dir: Optional[str] = None  # Path to OV IR shard dir (uses OVShard if set)


@dataclass
class PipelineConfig:
    """Full pipeline configuration."""
    model_id: str                          # HuggingFace model ID or local path
    model_local_path: str                  # Pre-downloaded model path on nodes
    quantization: str = "int4"             # For logging/identification only
    stages: list[StageConfig] = field(default_factory=list)
    max_new_tokens: int = 256
    temperature: float = 0.0              # Greedy by default for reproducibility
    shard_base_dir: Optional[str] = None   # Base dir for OV IR shards (stage_N/ subdirs)

    @property
    def total_stages(self) -> int:
        return len(self.stages)

    @property
    def total_layers(self) -> int:
        if not self.stages:
            return 0
        return self.stages[-1].layer_end


@dataclass
class EvalConfig:
    """Evaluation harness configuration."""
    prompt_file: str = "cascadia/eval/prompts/short_chat.jsonl"
    max_prompts: int = 50
    max_new_tokens: int = 256
    warmup_prompts: int = 3
    repetitions: int = 3


@dataclass
class EvalResult:
    """Results from a single evaluation run."""
    tokens_per_sec_mean: float
    tokens_per_sec_std: float
    ttft_ms_p50: float
    ttft_ms_p95: float
    total_tokens_generated: int
    total_time_sec: float
    per_token_latencies_ms: list[float] = field(default_factory=list)
    network_time_ms: float = 0.0      # Total time spent on activation transfers
    compute_time_ms: float = 0.0      # Total time spent on forward passes


def load_pipeline_config(path: str) -> PipelineConfig:
    """Load pipeline config from JSON."""
    import os
    with open(path) as f:
        data = json.load(f)
    stages = [StageConfig(**s) for s in data.pop("stages")]
    config = PipelineConfig(stages=stages, **data)

    # Populate shard_dir on each stage from shard_base_dir if not set explicitly
    if config.shard_base_dir:
        for i, stage in enumerate(config.stages):
            if stage.shard_dir is None:
                stage.shard_dir = os.path.join(config.shard_base_dir, f"stage_{i}")

    return config
