from ..config import Config
from .base import BaseModel
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
from huggingface_hub import hf_hub_download

class LCPModel(BaseModel):
    def __init__(
        self, 
        config: Config,
        repo_id="jartine/llava-v1.5-7B-GGUF",
        clip_model="llava-v1.5-7b-mmproj-Q4_0.gguf",
        model="llava-v1.5-7b-Q4_K.gguf"
    ):
        self.config = config
        clip_path = hf_hub_download(repo_id=repo_id, filename=clip_model)
        self.model_path = hf_hub_download(repo_id=repo_id, filename=model)

        self.llava = Llama(
            model_path=self.model_path,
            chat_handler=Llava15ChatHandler(
                clip_model_path=clip_path, 
                verbose=config.sys_logging
            ),
            max_tokens=150,
            n_ctx=4096,
            n_gpu_layers=config.n_gpu_layers,
            logits_all=True,
            temperature=config.temperature,
            verbose=config.sys_logging
        )

        print(f"Llama C++ model: {model} initialized\n")

    def process_image(self, text: str, image_path: str) -> str:
        image_uri = self.image_to_base64_data_uri(image_path)
        
        output = self.llava.create_chat_completion(
            messages=[
                {
                    "role": "system", 
                    "content": "You are an assistant who describes images exactly as instructed"
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_uri}},
                        {"type": "text", "text": text}
                    ]
                }
            ]
        )

        response = output['choices'][0]['message']['content']
        
        if self.config.logging:
            print(f"\nLLAVA OUTPUT: {response}\n")
            
        return self.strip_text(response)