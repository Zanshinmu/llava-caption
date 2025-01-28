from abc import ABC, abstractmethod
from pathlib import Path
import base64

class BaseModel(ABC):
    """Abstract base class for all caption models."""
    DEFAULT_CAPTION_PROMPT = """accurately and concisely describe the visual elements present in the image, including objects, people, colors, and spatial relationships, focusing only on what is visible, without embellishment or metaphor"""
    def __init__(self, config):
        self.config = config
    
    @abstractmethod
    def process_image(self, text: str, image_path: str) -> str:
        """Process an image and generate a caption."""
        pass

    @abstractmethod
    def direct_caption(self, image_path: str, prompt: str = DEFAULT_CAPTION_PROMPT) -> str:
        """Generate a caption directly from image."""
        pass
    
    def image_to_base64_data_uri(self, file_path: str) -> str:
        """Convert image file to base64 data URI.
        
        Args:
            file_path: Path to image file
            
        Returns:
            Base64 data URI string
        """
        with open(file_path, "rb") as img_file:
            base64_data = base64.b64encode(img_file.read()).decode('utf-8')
        return f"data:image/png;base64,{base64_data}"
    
    @staticmethod 
    def strip_text(text: str) -> str:
        """Clean up generated text.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text string
        """
        # Remove newlines and extra whitespace
        return " ".join(text.split())
