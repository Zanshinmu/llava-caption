from ..config import Config
from .base import BaseModel
import httpx
from ollama import Client, generate
import subprocess
from tqdm import tqdm

class OLModel(BaseModel):
    def __init__(self, config: Config, model="llava:7b-v1.5-q4_K_S"):
        self.config = config
        self.model = model
        self.client = Client(host=f"http://{config.ollama_address}")
        
        print(f"\nInitializing Ollama Processor\n")
        try:
            self.client.show(self.model)
        except httpx.ConnectError:
            self._start_ollama()
        except Exception as e:
            if getattr(e, 'status_code', None) == 404:
                self._pull_model()
        else:
            print(f"Connected to Ollama:{config.ollama_address}:{self.model}\n")

    def _start_ollama(self):
        process = subprocess.run(['ollama', 'list'])
        if process.returncode != 0:
            raise RuntimeError(f"Error starting ollama: {process.stderr}")

    def _pull_model(self):
        print(f"Pulling {self.model}")
        current_digest, bars = '', {}
        
        for progress in self.client.pull(self.model, stream=True):
            digest = progress.get('digest', '')
            
            if digest != current_digest and current_digest in bars:
                bars[current_digest].close()
                
            if not digest:
                print(progress.get('status'))
                continue
                
            if digest not in bars and (total := progress.get('total')):
                bars[digest] = tqdm(total=total, desc=f'pulling {digest[7:19]}', 
                                  unit='B', unit_scale=True)
                
            if completed := progress.get('completed'):
                bars[digest].update(completed - bars[digest].n)
                current_digest = digest

    def process_image(self, text: str, image_path: str) -> str:
        instruct = f"{self.config.system_prompt}'{text}'"
        settings = {"num_predict": 150, "temperature": self.config.temperature}
        
        response = self.client.generate(
            self.model,
            instruct,
            images=[image_path],
            options=settings
        )["response"]
        
        return self.strip_text(response)

    def llm_completion(self, system: str, text: str, label: str, json_format: bool = False) -> str:
        settings = {"num_predict": 1024, "seed": 31337, "temperature": 0.0}
        
        if json_format:
            instruct = f"Respond only in JSON with the response string named 'response':{system}\n {label}'{text}'"
            response = self.client.generate(
                self.model, 
                instruct, 
                options=settings, 
                format="json"
            )["response"]
        else:
            instruct = f"{system}\n {label}'{text}'"
            response = self.client.generate(
                self.model,
                instruct,
                options=settings
            )["response"]

        return response