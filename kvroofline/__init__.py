"""kvroofline: roofline-guided design-space exploration for KV-cache-bound long-context LLM decode."""
from .compression import Compression
from .engine import Calibration, SramPolicy, crossover_S, evaluate, max_batch, resolve_batch
from .hardware import PRESETS, Hardware, get_hw
from .models import MODELS, ModelConfig, get_model
from .layerwise import LayerMetrics, analyze_layerwise, layerwise_dataframe

__all__ = ["Compression", "Calibration", "SramPolicy", "crossover_S", "evaluate", "max_batch", "resolve_batch",
           "PRESETS", "Hardware", "get_hw", "MODELS", "ModelConfig", "get_model",
           "LayerMetrics", "analyze_layerwise", "layerwise_dataframe"]
