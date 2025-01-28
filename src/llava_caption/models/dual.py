from ..config import Config
from .base import BaseModel
from .ollama import OLModel
from .llama_cpp import LCPModel
import pandas as pd
import string
import json_repair
from tqdm import tqdm

class DualModel(BaseModel):
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize LLaVA 13B for vision
        llava_repo = "PsiPi/liuhaotian_llava-v1.5-13b-GGUF"
        self.llava = LCPModel(
            config,
            repo_id=llava_repo,
            clip_model="mmproj-model-Q4_0.gguf",
            model="llava-v1.5-13b-Q4_0.gguf"
        )
        
        # Initialize Mixtral for text processing
        self.llm = OLModel(config, model="mixtral:8x7b-instruct-v0.1-q5_0")

    def _identify_subject(self, element: str, context: str) -> str:
        instruction = (
            "Follow instructions and respond in json.\n"
            f"1. Find the owner of '{element}' in the context text. Is it a man, woman, object or background?\n"
            f"2. Use 2 words or less from the context text to identify the owner of {element}.\n"
            f"3. Respond with only the requested result. Return only two words in json response.\n"
        )
        
        response = self.llm.llm_completion(instruction, context, "context:", json_format=True)
        json_data = json_repair.loads(response)
        
        if self.config.logging:
            print(f"Subject of '{element}' is {json_data['response']}\n")
            
        return json_data['response']

    def _elements_completion(self, text: str) -> pd.DataFrame:
        instruction = (
            "Follow these instructions to complete the task:\n"
            "1. Compile a list of comma-separated elements from the element text.\n"
            "2. Organize the elements by subject and description or action of that subject.\n"
            "3. Use identical words to element text. Use 'man' or 'woman' instead of a name.\n"
            "4. Each property, action, or description of a subject must be a separate element.\n"
            "5. Each element must only be added once. Do not number the list.\n"
            "7. Respond only with the comma-separated list of elements.\n"
        )
        
        response = self.llm.llm_completion(instruction, text, "Element text:")
        elements = response.replace("\n", "").split(',')
        
        if self.config.logging:
            print(f"\nElements:\n{elements}\n")
            
        return pd.DataFrame(elements, columns=['Element'])

    def _questions_completion(self, elements: pd.DataFrame) -> pd.DataFrame:
        context = ', '.join(elements['Element'])
        instruction = (
            "Perform the task without remarks.\n"
            "Adhere strictly to these guidelines:\n"
            "1. Process the element text into a simple, direct question relating to the subject.\n"
            "2. Example question: 'Is <subject> <element text> in the image?'\n"
            "3. The question must require a yes/no answer to verify element is visible in an image\n"
            "4. Use exact element text to create the question. Try to use correct grammar.\n"
            "5. Use the subject and the element text for the question.\n"
            "6. Respond only with a simple question.\n"
        )
        
        print("Generating Questions from Elements.\n")
        questions = []
        
        for idx in tqdm(elements.index):
            element = elements.iloc[idx]["Element"]
            subject = self._identify_subject(element, context)
            
            response = self.llm.llm_completion(
                instruction, 
                element, 
                f"subject: {subject} element text:", 
                json_format=True
            )
            
            json_data = json_repair.loads(response)
            questions.append(json_data['response'])
            
        elements['Question'] = questions
        return elements

    def _query_llava(self, questions: pd.DataFrame, image_path: str) -> str:
        llava_response = []
        print("Querying Llava model with questions.\n")
        
        for idx in tqdm(questions.index):
            question = questions.iloc[idx]['Question']
            element = questions.iloc[idx]['Element']
            
            answer = self.llava.process_image(
                f"Answer accurately with yes/no: {question}:{element}?", 
                image_path
            )
            
            if self.config.logging:
                print(f"Results: {idx}\n {question}\n {element}\n {answer}\n")
                
            if "Yes" in answer:
                llava_response.append(element)
                
        return ", ".join(llava_response)

    def _caption_completion(self, visible: str) -> str:
        instruction = (
            "Follow these instructions to complete the task:\n"
            "1. Process the text into elements for an image caption.\n"
            "2. Organize the elements of the text into a logical structure for image description.\n"
            "3. Put the subject of the image first with related description.\n"
            "4. Background elements and elements not related to the subject come later.\n"
            "5. Do not modify the text or add anything, just change the structure.\n"
            "6. Do not leave any of the text elements out or modify them.\n"
            "7. If the subject is a person, use appropriate pronouns and nouns.\n"
            "8. Make sure the list is comma-separated and not numbered.\n"
        )
        
        response = self.llm.llm_completion(instruction, visible)
        
        if self.config.logging:
            print(f"Caption: {response}\n")
            
        return response

    def process_image(self, text: str, image_path: str) -> str:
        elements = self._elements_completion(text)
        questions = self._questions_completion(elements)
        visible = self._query_llava(questions, image_path)
        visible = self.strip_text(visible)
        
        if self.config.secondary_caption:
            return self._caption_completion(visible)
        
        return visible
    
    def llm_completion(self, instruction, image_path):
        response = self.llm.llm_completion(instruction, image_path, "response")
        return response
    
    def direct_caption(self, image_path, instruction=BaseModel.DEFAULT_CAPTION_PROMPT):
        response = self.llm_completion(instruction, image_path)
        return response
