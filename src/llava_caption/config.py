import os
import ast
from dataclasses import dataclass
from typing import Optional, Tuple

def parse_bool_env(env_var: str, default: bool) -> bool:
    """Parse boolean from environment variable with fallback to default."""
    value = os.environ.get(env_var)
    if value is None:
        return default
    try:
        return bool(ast.literal_eval(value))
    except (ValueError, SyntaxError):
        return default

def parse_host_port(addr: str) -> Tuple[str, int]:
    """Parse host:port string into separate host and port."""
    if ':' in addr:
        host, port = addr.split(':', 1)
        return host, int(port)
    return addr, 11434  # Default Ollama port

@dataclass
class Config:
    # Settings from command line args, with fallback to env vars
    model: str = os.environ.get('LLAVA_PROCESSOR', "MLXModel")
    system_prompt: str = "Describe the image following this style:"
    temperature: float = float(os.environ.get('TEMPERATURE', "0.1"))
    n_gpu_layers: int = int(os.environ.get('N_GPU_LAYERS', "-1"))
    preprocessor: bool = parse_bool_env('PREPROCESSOR', True)
    secondary_caption: bool = parse_bool_env('SECONDARY_CAPTION', False)
    sys_logging: bool = parse_bool_env('SYS_LOGGING', False)
    logging: bool = parse_bool_env('LOGGING', False)
    ollama_address: str = os.environ.get('OLLAMA_REMOTEHOST', "127.0.0.1:11434")
    direct_caption: bool = False

    @property
    def ollama_host(self) -> str:
        """Get Ollama host from address."""
        return parse_host_port(self.ollama_address)[0]

    @property
    def ollama_port(self) -> int:
        """Get Ollama port from address."""
        return parse_host_port(self.ollama_address)[1]
