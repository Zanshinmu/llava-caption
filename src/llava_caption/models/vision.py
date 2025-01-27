from ..config import Config
from .base import BaseModel
from pathlib import Path
import base64
from ollama import generate
import json_repair
import string

class VisionModel(BaseModel):
    def __init__(self, config: Config):
        self.config = config
        self.model = "llama3.2-vision:11b-instruct-q8_0"
        self.options = {
            'temperature': config.temperature,
            'num_predict': 160
        }

    def _llm_completion(
        self, 
        prompt: str, 
        image_path: str, 
        format: str = 'json',
        system: str = "You are an image captioning assistant who accurately and concisely describes the visual elements present in an image, including objects, colors, and spatial relationships, focusing only on what is visible, without embellishment or metaphor. You never refer to the image directly, you only describe the contents. You do not interpret the image, you describe it objectively."
    ) -> str:
        image = base64.b64encode(Path(image_path).read_bytes()).decode()
        response = generate(
            self.model,
            prompt,
            images=[image],
            stream=False,
            format=format,
            options=self.options,
            system=system
        )
        return response['response']

    def _secondary_completion(self, prompt: str, image_path: str) -> str:
        instruction = f"Generate a simple caption using the following text to guide your description. Check against the image to insure accuracy:'{prompt}'\n"
        caption = self._llm_completion(instruction, image_path, format='')
        return self.strip_text(caption)

    def _caption_completion(self, prompt: str, image_path: str) -> str:
        cleaned_prompt = self.strip_text(prompt)
        instruction = (
            f"Compare the following text with the image and remove non-visible elements: '{cleaned_prompt}'\n"
            f"Create a JSON object named 'text' containing a single string with the revised text.\n"
            f"Be concise and use the same words and style as the text.\n"
        )
        
        if self.config.logging:
            print(f"\nInstruction:\n{instruction}\n")
            
        response = self._llm_completion(instruction, image_path)
        
        if self.config.logging:
            print(f"\nElements:\n{response}\n")
            
        return response

    def process_image(self, text: str, image_path: str) -> str:
        caption = self._caption_completion(text, image_path)
        json_data = json_repair.loads(caption)
        
        try:
            caption = json_data['text']
        except KeyError:
            caption = self._caption_completion(text, image_path)
            json_data = json_repair.loads(caption)
            caption = json_data['text']
            
        if self.config.secondary_caption:
            return self._secondary_completion(caption, image_path)
            
        return caption