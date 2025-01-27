"""
LLaVA Caption - Tool for automatically captioning images using various LLaVA models
"""

from .config import Config
from .models.base import BaseModel
from .models.ollama import OLModel
from .models.huggingface import HFModel
from .models.llama_cpp import LCPModel
from .models.dual import DualModel
from .models.vision import VisionModel
from .models.mlx import MLXModel
from .utils.image import resize_and_save_image
from .utils.text import preprocess_text

__version__ = "0.7.0"
__author__ = "David \"Zanshinmu\" Van de Ven"
__description__ = "Tool for automatically captioning images using various LLaVA models"

__all__ = [
    'Config',
    'BaseModel',
    'OLModel',
    'HFModel',
    'LCPModel',
    'DualModel',
    'VisionModel',
    'MLXModel',
    'resize_and_save_image',
    'preprocess_text',
]