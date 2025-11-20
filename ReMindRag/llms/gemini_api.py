from .base import AgentBase
import vertexai
from vertexai.generative_models import GenerativeModel, SafetySetting, HarmCategory, HarmBlockThreshold
from typing import Optional, List, Dict, Any

class GeminiAgent(AgentBase):
    def __init__(self, project_id: str, location: str, model_name: str = "gemini-2.5-flash"):
        # THIS IS THE TRIGGER: vertexai.init() automatically finds your gcloud credentials
        self.project_id = project_id
        self.location = location
        vertexai.init(project=project_id, location=location)
        
        self.model_name = model_name
        self.model = GenerativeModel(model_name)
        
        # Configure safety settings (important for processing books freely)
        self.safety_config = [
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH
            ),
        ]

    def generate_response(self, system_prompt: Optional[str], chat_history: List[Dict[str, Any]]) -> str:
        # Convert ReMindRAG's chat history format to a simple prompt for Gemini
        full_prompt = ""
        if system_prompt:
            full_prompt += f"System Instructions: {system_prompt}\n\n"
            
        for msg in chat_history:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            full_prompt += f"{role}: {content}\n"
            
        full_prompt += "ASSISTANT:"

        try:
            response = self.model.generate_content(
                full_prompt,
                generation_config={"temperature": 0},
                safety_settings=self.safety_config
            )
            return response.text
        except Exception as e:
            print(f"Gemini Error: {e}")
            return "Error generating response."
