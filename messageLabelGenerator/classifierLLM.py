import os
import openai


class AssistantError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class BaseAssistant:
    def __init__(self,
                 system_prompt: str,
                 json_schema: dict = None
                 ):
        if json_schema is None:
            json_schema = {"type": "text"}
        openai.api_key = os.getenv('OPENAI_API_KEY')
        self.system_prompt = system_prompt
        self.json_schema = json_schema

    def respond(self, user_input: str) -> str:
        try:
            response = openai.responses.create(
                prompt={
                    "id": self.system_prompt,
                },
                input=user_input,
                text={"format": self.json_schema}
            )
            if not response.output_text:
                raise AssistantError('Ответ от ассистента пустой')
            return response.output_text
        except Exception as e:
            raise AssistantError(f'Ошибка при получении ответа: {e!r}')


class ClassifierAssistant(BaseAssistant):
    def __init__(self):
        system_prompt = "pmpt_68e8b47991f0819094f05d51eb5780a10ff88c389be96726"
        json_schema = None
        super().__init__(system_prompt, json_schema)

    def run(self, user_input: str):
        output = self.respond(user_input)
        return output
