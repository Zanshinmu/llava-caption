"""
Model implementations for LLaVA Caption tool.
Provides various model classes for image captioning using different backends.
"""

from .base import BaseModel
from .ollama import OLModel
from .huggingface import HFModel
from .llama_cpp import LCPModel
from .dual import DualModel
from .vision import VisionModel
from .mlx import MLXModel

__all__ = [
    'BaseModel',
    'OLModel',
    'HFModel',
    'LCPModel',
    'DualModel',
    'VisionModel',
    'MLXModel',
]

# Map of model names to their implementations
MODEL_MAP = {
    'OLModel': OLModel,
    'HFModel': HFModel,
    'LCPModel': LCPModel,
    'DualModel': DualModel,
    'VisionModel': VisionModel,
    'MLXModel': MLXModel
}