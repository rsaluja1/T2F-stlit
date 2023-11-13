import os
from utils import read_yaml_file, read_pdf

prompt_path = "prompts/talk_to_file_prompts copy.yaml"
dirname = os.path.dirname(os.path.abspath(__file__))
prompt_path_ttf = os.path.join(dirname, prompt_path)


def prompt_creator(file_text: str, question: str):
    
    prompt = read_yaml_file(prompt_path_ttf)["TALK_TO_FILE"]
    # prompt = prompt.replace("<<file_text>>", f"{file_text}").replace(
    #         "<<user_question>>", f"{question}"
    #     )
    for item in prompt:
        if 'content' in item:
            item['content'] = item['content'].replace('<<file_text>>', f"{file_text}")
            item['content'] = item['content'].replace('<<user_question>>',f"{question}")
            
    return prompt


if __name__ == "__main__":
    # file_text = input("Enter the Text of the File you wish to Chat with ")
    # question = input("Enter your Question ")


    file_text = read_pdf("temp_data/NPL- D2 Document (1).pdf")
    question = "Who is the author?" 

    print(prompt_creator(file_text,question))