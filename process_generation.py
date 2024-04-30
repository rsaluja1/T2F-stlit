import openai
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
import os
from dotenv import load_dotenv
from fastapi import HTTPException

from hyperparams_handler import HyperparamsHandler
from prompt_creator import PromptCreator
from t2f_router import get_route_name
from utils import token_counter

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class GenerativeLayer:
    """Class to handle the generative layer of the application."""

    @staticmethod
    async def create_non_stream_llm_call(hp, prompt):
        """Create a subscription to the OpenAI API."""

        openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

        params = {
            "messages": prompt,
            "temperature": hp["temperature"],
            "max_tokens": hp["max_tokens"],
            "top_p": hp["top_p"],
            "frequency_penalty": hp["frequency_penalty"],
            "presence_penalty": hp["presence_penalty"],
            "stop": hp["stop_sequences"],
            "stream": False,
            "model": hp["model_name"]
        }

        try:
            return await openai_client.chat.completions.create(**params)
        except openai.OpenAIError as e:
            raise HTTPException(status_code=500, detail=f"OpenAI call failed: {str(e)}")

    @staticmethod
    async def create_anthropic_non_stream_subscription(hp, prompt):
        """Create a non-streaming subscription to the ANTHROPIC API."""

        anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

        params = {
            "model": hp["model"],
            "max_tokens": hp["max_tokens"],
            "temperature": hp["temperature"],
            "system": prompt.pop(0)["content"],
            "messages": prompt,
            "stop_sequences": hp["stop_sequences"],
        }

        try:
            anthropic_q_ans = await anthropic_client.messages.create(**params)
            return anthropic_q_ans.content

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Anthropic call failed: {str(e)}")

    @staticmethod
    async def process_get_anthropic_answer(file_id: str, file_name: str, file_text: str, question: str) -> str:
        """Process a generative layer request for an Answer from Anthropic using the full file_text & question"""

        hp, prompt = (HyperparamsHandler.handle_get_anthropic_answer(),
                      PromptCreator.create_get_anthropic_answer(file_text, question))

        token_count = token_counter(str(prompt), model_name="gpt-4")

        if token_count > 199000:
            raise HTTPException(
                status_code=422,
                detail="Token limit Exceeded. Your File is too big. Please try again with a smaller file.",
                headers={
                    "X-Error": "Token limit Exceeded. Your File is too big. Please try again with a smaller file."},
            )

        if not hp or not prompt:
            raise HTTPException(
                status_code=500, detail=f"Invalid Question: {question}"
            )

        response = await GenerativeLayer.create_anthropic_non_stream_subscription(
            hp, prompt
        )

        anthropic_answer = ("Document name: " + file_name + "\n\n"
                            + "Document file id: " + file_id + "\n\n"
                            + "Answer Text: " + response[0].text + "\n\n"
                            + "----END OF ANSWER----")

        return anthropic_answer

    @staticmethod
    async def process_ttmf_final_answer(anthropic_answers: str, question: str) -> str:
        """Process a generative layer request for a Final Answer from GPT-4 using
        the Anthropic Answers & user question"""

        route_name = get_route_name(question)

        hp, prompt = (
            HyperparamsHandler.handle_ttmf_final_answer(),
            PromptCreator.create_ttmf_final_answer(anthropic_answers, question, route_name),
        )

        token_count = token_counter(str(prompt), model_name="gpt-4")

        if token_count > 127000:
            raise HTTPException(
                status_code=422,
                detail="Token limit Exceeded. Your File is too big. Please try again with a smaller file.",
                headers={
                    "X-Error": "Token limit Exceeded. Your File is too big. Please try again with a smaller file."},
            )

        if not hp or not prompt:
            raise HTTPException(
                status_code=500,
                detail=f"Invalid Final Answer Generation Request "
                       f"for {anthropic_answers}",
            )

        response = await GenerativeLayer.create_non_stream_llm_call(hp, prompt)
        return response.choices[0].message.content
