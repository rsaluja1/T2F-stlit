import os
from utils import read_yaml_file, token_counter

parent_prompt_path = "prompts"
dirname = os.path.dirname(os.path.abspath(__file__))
#prompt_path_ttf = os.path.join(dirname, parent_prompt_path)


def prompt_creator(route_name: str, user_query: str, retreived_chunks: str = None):
    
    if route_name == "chitchat" or route_name == "gratitude":

        prompt_file = "talk_to_file_chitchat_anthropic.yaml"
        prompt_path = os.path.join(dirname, parent_prompt_path, prompt_file)

        prompt_system = read_yaml_file(prompt_path)["TALK_TO_FILE_CHITCHAT"]["SYSTEM"]
        prompt_messages = read_yaml_file(prompt_path)["TALK_TO_FILE_CHITCHAT"]["MESSAGES"]

        for item in prompt_messages:
            if 'content' in item:
                item['content'] = item['content'].replace('<<user_query>>',f"{user_query}")

        return prompt_system, prompt_messages
    
    else:
        prompt_file = "talk_to_file_new_anthopic.yaml"
        prompt_path = os.path.join(dirname, parent_prompt_path, prompt_file)
        
        prompt_system = read_yaml_file(prompt_path)["TALK_TO_FILE_REFERENCING"]["SYSTEM"]
        prompt_messages = read_yaml_file(prompt_path)["TALK_TO_FILE_REFERENCING"]["MESSAGES"]
        # prompt = prompt.replace("<<file_text>>", f"{file_text}").replace(
        #         "<<user_question>>", f"{question}"
        #     )
        #retreived_chunks = retreiver(question)
        chunk_tokens = token_counter(retreived_chunks,model_name="gpt-4")

        for item in prompt_messages:
            if 'content' in item:
                item['content'] = item['content'].replace('<<retrieved_chunks>>', f"{retreived_chunks}")
                item['content'] = item['content'].replace('<<user_question>>',f"{user_query}")
                
        return prompt_system, prompt_messages, chunk_tokens


#if __name__ == "__main__":
    # file_text = input("Enter the Text of the File you wish to Chat with ")
    # question = input("Enter your Question ")


    # file_text = read_pdf("temp_data/managing_ai_risks.pdf")
    # question = "Who is the author?" 

    # print(prompt_creator(question))

    # file_text = read_pdf("temp_data/managing_ai_risks.pdf")
    # # question = "Who is the author?"