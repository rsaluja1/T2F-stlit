import os
from utils import read_yaml_file

parent_prompt_path = "prompts"
dirname = os.path.dirname(os.path.abspath(__file__))
# prompt_path_ttf = os.path.join(dirname, parent_prompt_path)


def prompt_creator(route_name: str, file_text: str):
    if route_name == "chitchat" or route_name == "gratitude":
        prompt_file = "talk_to_file_chitchat_gemini.yaml"
        prompt_path = os.path.join(dirname, parent_prompt_path, prompt_file)

        prompt_system, prompt_messages = (
            read_yaml_file(prompt_path)["TALK_TO_FILE_CHITCHAT"]["SYSTEM_PROMPT"],
            read_yaml_file(prompt_path)["TALK_TO_FILE_CHITCHAT"]["MESSAGES"],
        )

        for item in prompt_messages:
            item_list = item["parts"]
            for elem in item_list:
                elem["text"] = elem["text"].replace("<<file_text>>", f"{file_text}")

        return prompt_system, prompt_messages

    else:
        prompt_file = "talk_to_multiple_files_gemini.yaml"
        prompt_path = os.path.join(dirname, parent_prompt_path, prompt_file)

        prompt_system, prompt_messages = (
            read_yaml_file(prompt_path)["TALK_TO_MULTIPLE_FILES_GEMINI"][
                "SYSTEM_PROMPT"
            ],
            read_yaml_file(prompt_path)["TALK_TO_MULTIPLE_FILES_GEMINI"]["MESSAGES"],
        )

        for item in prompt_messages:
            item_list = item["parts"]
            for elem in item_list:
                elem["text"] = elem["text"].replace("<<file_text>>", f"{file_text}")

        return prompt_system, prompt_messages


# if __name__ == "__main__":
#     route_name = None
#     file_text = input("Enter the Text of the File you wish to Chat with ")
#     # question = input("Enter your Question ")

#     print(prompt_creator(route_name, file_text))


# #     file_text = read_pdf("temp_data/managing_ai_risks.pdf")
# #     question = "Who is the author?"

# #     print(prompt_creator(question))

# #     file_text = read_pdf("temp_data/managing_ai_risks.pdf")
# #     # question = "Who is the author?"
