from abc import ABC, abstractmethod
from pathlib import Path
import base64

class BaseModel(ABC):
    """Abstract base class for all caption models."""
    
    def __init__(self, config):
        self.config = config
    
    @abstractmethod
    def process_image(self, text: str, image_path: str) -> str:
        """Process an image and generate a caption.
        
        Args:
            text: The prompt text to guide caption generation
            image_path: Path to the image file
            
        Returns:
            Generated caption string
        """
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