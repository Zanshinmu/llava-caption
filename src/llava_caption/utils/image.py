from PIL import Image
import tempfile
from pathlib import Path

def resize_and_save_image(image_path: str) -> str:
    """Resize image within model limits while preserving aspect ratio."""
    max_size = (670, 670)
    
    image = Image.open(image_path)
    image.thumbnail(max_size, Image.LANCZOS)
    
    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    image.save(temp_file)
    image.close()
    
    return temp_file.name