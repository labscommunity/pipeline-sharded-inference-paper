"""Model sharding — split a model into layer ranges for pipeline stages."""

from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class ShardPlan:
    """Plan for splitting a model across pipeline stages."""
    model_id: str
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    vocab_size: int
    # Per-stage layer assignments: list of (start, end) tuples
    stage_layers: list[tuple[int, int]]

    @classmethod
    def create(cls, model_id: str, num_stages: int, layer_ratios: list[float] = None):
        """Create a shard plan for a model.

        Args:
            model_id: HuggingFace model ID or local path
            num_stages: Number of pipeline stages
            layer_ratios: Optional per-stage ratios (must sum to 1.0).
                         If None, layers are split uniformly.
        """
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

        num_layers = config.num_hidden_layers
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        vocab_size = config.vocab_size

        if layer_ratios is None:
            # Uniform split
            layer_ratios = [1.0 / num_stages] * num_stages

        assert len(layer_ratios) == num_stages
        assert abs(sum(layer_ratios) - 1.0) < 1e-6

        # Assign layers proportionally
        stage_layers = []
        cursor = 0
        for i, ratio in enumerate(layer_ratios):
            if i == num_stages - 1:
                # Last stage gets remaining layers
                stage_layers.append((cursor, num_layers))
            else:
                count = max(1, round(num_layers * ratio))
                stage_layers.append((cursor, cursor + count))
                cursor += count

        return cls(
            model_id=model_id,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            vocab_size=vocab_size,
            stage_layers=stage_layers,
        )

    def describe(self) -> str:
        lines = [f"ShardPlan for {self.model_id}"]
        lines.append(f"  Total layers: {self.num_layers}, hidden: {self.hidden_size}")
        for i, (start, end) in enumerate(self.stage_layers):
            lines.append(f"  Stage {i}: layers [{start}, {end}) — {end - start} layers")
        return "\n".join(lines)
