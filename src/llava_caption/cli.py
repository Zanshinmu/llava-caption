import argparse
from pathlib import Path
from typing import Type
import sys
import os

from .config import Config
from .models.base import BaseModel
from .utils.text import preprocess_text
from .utils.image import resize_and_save_image

def parse_args() -> tuple[Path, Config]:
    parser = argparse.ArgumentParser(description='Caption images using various ML models')
    
    # Create default config from env vars
    default_config = Config()
    
    parser.add_argument('directory', nargs='?', default='.',
                      help='Directory containing images and text files')
    
    parser.add_argument('--model', default=default_config.model,
                      choices=['OLModel', 'HFModel', 'LCPModel', 'DualModel', 'VisionModel', 'MLXModel'],
                      help='Model to use for captioning (env: LLAVA_PROCESSOR)')
                      
    parser.add_argument('--temperature', type=float, default=default_config.temperature,
                      help='Model temperature (env: TEMPERATURE)')
                      
    parser.add_argument('--gpu-layers', type=int, default=default_config.n_gpu_layers,
                      help='Number of GPU layers (-1 for all) (env: N_GPU_LAYERS)')
                      
    parser.add_argument('--no-preprocess', action='store_false', dest='preprocessor',
                      default=default_config.preprocessor,
                      help='Disable text preprocessing (env: PREPROCESSOR)')
                      
    parser.add_argument('--secondary-caption', action='store_true',
                      default=default_config.secondary_caption,
                      help='Enable secondary captioning (env: SECONDARY_CAPTION)')
                      
    parser.add_argument('--logging', action='store_true',
                      default=default_config.logging,
                      help='Enable detailed logging (env: LOGGING)')
                      
    parser.add_argument('--sys-logging', action='store_true',
                      default=default_config.sys_logging,
                      help='Enable system logging (env: SYS_LOGGING)')
                      
    parser.add_argument('--ollama-address', default=default_config.ollama_address,
                      help='Ollama address in host:port format (env: OLLAMA_REMOTEHOST)')
    
    parser.add_argument('--direct-caption', action='store_true',
                   help='Directly caption images without prompt comparison')

    args = parser.parse_args()
    
    config = Config(
        model=args.model,
        temperature=args.temperature,
        n_gpu_layers=args.gpu_layers,
        preprocessor=args.preprocessor,
        secondary_caption=args.secondary_caption,
        logging=args.logging,
        sys_logging=args.sys_logging,
        ollama_address=args.ollama_address,
        direct_caption=args.direct_caption
    )
    
    return Path(args.directory), config

def get_model_class(config: Config) -> Type[BaseModel]:
    """Get the model class based on configuration."""
    try:
        from .models import ollama, huggingface, llama_cpp, dual, vision, mlx
        
        model_map = {
            'OLModel': ollama.OLModel,
            'HFModel': huggingface.HFModel,
            'LCPModel': llama_cpp.LCPModel,
            'DualModel': dual.DualModel,
            'VisionModel': vision.VisionModel,
            'MLXModel': mlx.MLXModel
        }
        
        # Get clean model name for display
        model_name = config.model
        processor_name = model_name
        if model_name.endswith('Model'):
            processor_name = model_name[:-5]  # Remove 'Model' suffix
        print(f"\n<'{processor_name} processor loading'>\n")
        
        return model_map[config.model]
    except KeyError:
        sys.exit(f"\nError: Model class {config.model} does not exist!\n")

def process_directory(directory: Path, model: BaseModel, config: Config) -> None:
    """Process all files in directory and generate captions."""
    for root, dirs, files in os.walk(directory):
        if config.direct_caption:
            print(f"Direct Caption Mode\n")
            filecount = sum(1 for file in files if file.endswith('.png'))
            processed = 0
            
            for file in files:
                if file.endswith('.png'):
                    png_path = Path(root) / file
                    temp_image = resize_and_save_image(png_path)
                    
                    if temp_image:
                        response = model.direct_caption(temp_image)
                        processed += 1
                        
                        # Write to corresponding text file
                        txt_path = png_path.with_suffix('.txt')
                        with open(txt_path, 'w') as f:
                            f.write(response)
                            
                        print(f"{file}: {processed} of {filecount}\n{response}\n")
                        
                        # Cleanup temp file
                        Path(temp_image).unlink()
        else:
            filecount = sum(1 for file in files if file.endswith('.txt'))
            processed = 0
            
            for file in files:
                if file.endswith('.txt'):
                    txt_path = Path(root) / file
                    with open(txt_path, 'r') as f:
                        text = f.read()
                    
                    processed_text = preprocess_text(text) if config.preprocessor else text
                    png_path = txt_path.with_suffix('.png')
                    
                    if not png_path.exists():
                        print(f"No corresponding image for {file}")
                        continue
                        
                    temp_image = resize_and_save_image(png_path)
                    if temp_image:
                        response = model.process_image(processed_text, temp_image)
                        processed += 1
                        
                        with open(txt_path, 'w') as f:
                            f.write(response)
                            
                        print(f"{file}: {processed} of {filecount}\n{response}\n")
                        
                        # Cleanup temp file
                        Path(temp_image).unlink()

def main():
    directory, config = parse_args()
    model_class = get_model_class(config)
    model = model_class(config)
    process_directory(directory, model, config)

if __name__ == '__main__':
    main()
