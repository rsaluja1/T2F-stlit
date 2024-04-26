import os
from typing import Optional

from models.T2F import PromptCreatorModel
from utils import read_yaml_file

parent_prompt_path = "prompts/talk_to_multiple_files_anthropic.yaml"
dirname = os.path.dirname(os.path.abspath(__file__))


class PromptCreator:
    """Class to handle the creation of prompts."""

    @staticmethod
    def _read_and_replace_content(values: PromptCreatorModel, prompt_path: str, route_name: Optional[str] = None):

        prompt_path_ttf = os.path.join(dirname, prompt_path)

        prompt = read_yaml_file(prompt_path_ttf)[values.prompt_key]
        for item in prompt:
            if "content" in item:
                if route_name is None:
                    item["content"] = item["content"].replace(values.base_text, f"{values.base_replace_text}")
                if values.replacement_key:
                    item["content"] = item["content"].replace(values.replacement_key, f"{values.replace_text}")

        return prompt

    @staticmethod
    def create_get_anthropic_answer(file_text: str, question: str):

        values = PromptCreatorModel(
                base_text="<<file_text>>",
                base_replace_text=file_text,
                prompt_key="GET_ANTHROPIC_ANSWER",
                replacement_key="<<user_question>>",
                replace_text=question
            )

        return PromptCreator._read_and_replace_content(values, parent_prompt_path)

    @staticmethod
    def create_ttmf_final_answer(anthropic_answers: str, question: str, route_name: str):

        if route_name == None:
            values = PromptCreatorModel(
                base_text="<<anthropic_answers>>",
                base_replace_text=anthropic_answers,
                prompt_key="TALK_TO_MULTIPLE_FILES",
                replacement_key="<<user_question>>",
                replace_text=question
            )

        else:
            values = PromptCreatorModel(
                prompt_key="TALK_TO_FILE_CHITCHAT",
                replacement_key="<<user_question>>",
                replace_text=question

            )

        return PromptCreator._read_and_replace_content(values, parent_prompt_path, route_name)

# def prompt_creator(route_name: str, file_text: str):
#     if route_name == "chitchat" or route_name == "gratitude":
#         prompt_file = "talk_to_file_chitchat_gemini.yaml"
#         prompt_path = os.path.join(dirname, parent_prompt_path, prompt_file)
#
#         prompt_system, prompt_messages = (
#             read_yaml_file(prompt_path)["TALK_TO_FILE_CHITCHAT"]["SYSTEM_PROMPT"],
#             read_yaml_file(prompt_path)["TALK_TO_FILE_CHITCHAT"]["MESSAGES"],
#         )
#
#         for item in prompt_messages:
#             item_list = item["parts"]
#             for elem in item_list:
#                 elem["text"] = elem["text"].replace("<<file_text>>", f"{file_text}")
#
#         return prompt_system, prompt_messages
#
#     else:
#         prompt_file = "talk_to_multiple_files_gemini.yaml"
#         prompt_path = os.path.join(dirname, parent_prompt_path, prompt_file)
#
#         prompt_system, prompt_messages = (
#             read_yaml_file(prompt_path)["TALK_TO_MULTIPLE_FILES_GEMINI"][
#                 "SYSTEM_PROMPT"
#             ],
#             read_yaml_file(prompt_path)["TALK_TO_MULTIPLE_FILES_GEMINI"]["MESSAGES"],
#         )
#
#         for item in prompt_messages:
#             item_list = item["parts"]
#             for elem in item_list:
#                 elem["text"] = elem["text"].replace("<<file_text>>", f"{file_text}")
#
#         return prompt_system, prompt_messages
