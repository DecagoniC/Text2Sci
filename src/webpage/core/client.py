import os
from google import genai
from google.genai import types

class AIClient:
    def __init__(self, prompt: str):
        self.prompt = prompt
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def generate(self, user_text: str) -> str:
        full_prompt = f"{self.prompt}\nПользователь: {user_text}\nОтвет:"
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[types.Part.from_text(text=full_prompt)]
        )
        return response.candidates[0].content.parts[0].text or "..."
