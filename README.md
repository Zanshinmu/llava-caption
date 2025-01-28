# LLaVA-Caption

By David "Zanshinmu" Van de Ven
zanshin.g1@gmail.com

Automatically caption images using various LLaVA multimodal models. This tool processes images with state-of-the-art vision language models to generate accurate, high-quality captions.

---

## Overview

LLaVA Caption was designed to solve a specific problem in AI training: when using generated images, the original prompts often contain elements that aren't present in the final images. Manual verification and captioning is time-consuming, but inaccurate captions make for bad training data. This tool provides higher quality captions than BLIP, with options ranging from basic processing to near-manual quality.

Llava Caption was built and tested on Apple Silicon. While cross-platform tools make it accessible to PCs, it hasn’t been tested on Linux or Windows. 

---

## Available Models

### MLXModel (Recommended)(Default)
- Uses Qwen2-VL-7B-Instruct-8bit with Apple's MLX framework
- Apple Silicon only
- Fast processing with 16GB unified memory
- Accuracy comparable to VisionModel
- **Requirements**: Apple Silicon Mac, 16GB+ unified memory

### VisionModel
- Uses Llama 3.2 Vision via Ollama
- High accuracy with moderate resource requirements
- Excellent results with secondary caption generation
- Ideal for training Flux/SD3
- **Requirements**: 24GB RAM, GPU recommended

### DualModel (Experimental)
- Combines LLaVA 1.5 and Mixtral
- Highest potential accuracy but resource-intensive
- Supports distributed processing across machines
- Currently experimental: may need optimization
- **Requirements**: 64GB RAM, GPU strongly recommended

### Additional Models
- **OLModel**: Basic Ollama-based processing
- **HFModel**: Hugging Face transformers-based processing (Note: MPS not supported on Apple Silicon)
- **LCPModel**: Direct LLaMA C++ processing

---

## Installation

### Prerequisites
- Python 3.10 ([python.org](https://www.python.org/downloads/))
- Git ([git-scm.com](https://git-scm.com/))
- Ollama ([ollama.com/download](https://ollama.com/download)) - Required for Ollama-based models

### Quick Start
```bash
# Clone repository
git clone https://github.com/yourusername/llava-caption.git
cd llava-caption

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .

# Verify installation
llava-caption --help
```

---

## Usage

### Basic Commands
```bash
# Basic usage with defaults
llava-caption /path/to/images/

# Specific model selection
llava-caption --model MLXModel /path/to/images/

# Direct captioning (no prompt comparison)
llava-caption --direct-caption /path/to/images/
```

### Command Line Options
```bash
llava-caption [OPTIONS] DIRECTORY

Arguments:
  DIRECTORY                      Directory containing images

Model Selection:
  --model MODEL                  Model to use (default: MLXModel)
                                [env: LLAVA_PROCESSOR]

Processing Modes:
  --direct-caption              Enable direct captioning mode
  --secondary-caption           Enable secondary captioning
                                [env: SECONDARY_CAPTION]
  --no-preprocess              Disable text preprocessing
                                [env: PREPROCESSOR]

Model Parameters:
  --temperature FLOAT           Generation temperature (default: 0.0)
                                [env: TEMPERATURE]
  --gpu-layers INT             GPU layers (-1 for all)
                                [env: N_GPU_LAYERS]

Ollama Configuration:
  --ollama-address HOST:PORT    Ollama address (default: 127.0.0.1:11434)
                                [env: OLLAMA_REMOTEHOST]

Logging:
  --logging                     Enable detailed logging
  --sys-logging                Enable system logging
    
  
```

### Example Usage Patterns
```bash
# MLXModel with direct captioning
llava-caption --model MLXModel --direct-caption /path/to/images/

# VisionModel with remote Ollama
llava-caption --model VisionModel --ollama-address 192.168.1.110:11434 /path/to/images/

# Secondary captioning with higher temperature
llava-caption --model VisionModel --secondary-caption --temperature 0.7 /path/to/images/

# Debug mode
llava-caption --logging --sys-logging /path/to/images/
```

---

## Important Notes

### PC Users
- You may need to remove any mlx entries from requirements.txt to install successfully. 

### Model Downloads
- Models are automatically downloaded via Hugging Face Hub or Ollama
- Initial downloads may take time and significant disk space
- Models are selected for optimal performance and resource usage

### Resource Requirements
- **CPU Mode**: Significant CPU and RAM usage, especially with HFModel
- **GPU Usage**: Set `TORCH_DEVICE="cuda:0"` for Nvidia GPU support
- **Distributed Processing**: Possible to run models across 2 hosts using DualModel

### File Handling
- Expects matching .png and .txt files in target directory
- Existing text files will be overwritten with new captions
- In direct caption mode, creates new .txt files for each image

---

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
