import os
import openai


class AssistantError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class BaseAssistant:
    def __init__(self,
                 system_prompt: str,
                 json_schema: dict = None):
        if json_schema is None:
            json_schema = {"type": "text"}
        self.system_prompt = system_prompt
        self.json_schema = json_schema
        openai.api_key = os.getenv('OPENAI_API_KEY')

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


class ChatClassifierAssistant(BaseAssistant):
    def __init__(self):
        system_prompt = "pmpt_68e8b88526f4819396be91ca2ca0eeb907bf75b775700bf1"
        json_schema = None

        super().__init__(system_prompt, json_schema)

    def run(self, chat_message: str):
        output = self.respond(chat_message)
        return output
