import os
from utils import read_yaml_file, read_pdf, token_counter

prompt_path = "prompts/T2F_Ref_Revised.yaml"
dirname = os.path.dirname(os.path.abspath(__file__))
prompt_path_ttf = os.path.join(dirname, prompt_path)


def prompt_creator(question: str, retreived_chunks: str):
    
    prompt = read_yaml_file(prompt_path_ttf)["TALK_TO_FILE_REFERENCING"]
    # prompt = prompt.replace("<<file_text>>", f"{file_text}").replace(
    #         "<<user_question>>", f"{question}"
    #     )
    #retreived_chunks = retreiver(question)
    chunk_tokens = token_counter(retreived_chunks,model_name="gpt-4")
    
    


    for item in prompt:
        if 'content' in item:
            item['content'] = item['content'].replace('<<retrieved_chunks>>', f"{retreived_chunks}")
            item['content'] = item['content'].replace('<<user_question>>',f"{question}")
            
    return prompt, chunk_tokens


# if __name__ == "__main__":
#     # file_text = input("Enter the Text of the File you wish to Chat with ")
#     # question = input("Enter your Question ")


#     file_text = read_pdf("temp_data/managing_ai_risks.pdf")
#     question = "Who is the author?" 

#     print(prompt_creator(question))