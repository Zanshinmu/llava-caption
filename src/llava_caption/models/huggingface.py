from ..config import Config
from .base import BaseModel
import torch
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

class HFModel(BaseModel):
    def __init__(self, config: Config, model_path="llava-hf/llava-v1.6-mistral-7b-hf"):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_tokens = 150
        
        self.dtype = (torch.float16 if self.device.type == "mps" 
                     else torch.bfloat16 if self.device.type == "cuda" 
                     else torch.float32)

        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            device_map=self.device,
            do_sample=True,
            temperature=config.temperature,
        )
        
        self.processor = LlavaNextProcessor.from_pretrained(model_path)
        self.processor.tokenizer.padding_side = "left"
        
        print(f"{model_path} initialized for device '{self.device}'\n")

    def _strip_inst_tags(self, text: str) -> str:
        try:
            return text.split('[/INST]')[1]
        except IndexError:
            return text

    def process_image(self, text: str, image_path: str) -> str:
        image = Image.open(image_path)
        prompt = f"[INST] <{image}>\n{self.config.system_prompt}'{text}'[/INST]"

        inputs = self.processor(
            image,
            prompt,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        self.model.config.image_token_index = 1
        output = self.model.generate(
            **inputs,
            max_new_tokens=self.max_tokens,
            pad_token_id=self.processor.tokenizer.pad_token_id
        )

        response = self._strip_inst_tags(
            self.processor.decode(output[0], skip_special_tokens=True)
        )
        
        image.close()
        return self.strip_text(response)
   
    def direct_caption(self, image_path, instruction=BaseModel.DEFAULT_CAPTION_PROMPT):
        response = self.process_image(image_path, instruction)
        return self.strip_text(response)
