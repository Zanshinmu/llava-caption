from ..config import Config 
from .base import BaseModel
import json_repair
import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config, load_image

class MLXModel(BaseModel):
    def __init__(self, config: Config):
        self.config = config
        self.model_path = "mlx-community/Qwen2-VL-7B-Instruct-8bit"
        self.model, self.processor = load(self.model_path)
        self.options = {
            'temperature': config.temperature,
            'num_predict': 160
        }

    def llm_completion(
        self,
        prompt: str,
        image_path: str,
        format: str = 'json',
        system: str = "You are an image captioning assistant who accurately and concisely describes the visual elements present in an image, including objects, colors, and spatial relationships, focusing only on what is visible, without embellishment or metaphor."
    ) -> str:
        image = load_image(image_path)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]
        
        formatted_prompt = apply_chat_template(
            self.processor,
            self.model.config,
            messages,
            num_images=1
        )
        
        result = generate(
            self.model,
            self.processor,
            formatted_prompt,
            image,
            temperature=self.options['temperature'],
            max_tokens=self.options['num_predict']
        )
        
        return result

    def _secondary_completion(self, prompt: str, image_path: str) -> str:
        instruction = f"Generate a simple caption using the following text to guide your description. Check against the image to ensure accuracy:'{prompt}'\n"
        caption = self.llm_completion(instruction, image_path, format='')
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
            
        response = self.llm_completion(instruction, image_path)
        
        if self.config.logging:
            print(f"\nElements:\n{response}\n")
            
        return response
        
    def direct_caption(self, image_path, instruction=BaseModel.DEFAULT_CAPTION_PROMPT):
        response = self.llm_completion(instruction, image_path)
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
