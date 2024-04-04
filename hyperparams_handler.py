import os
from utils import read_yaml_file, token_counter

hyperparams_path = "configs/hyperparams.yaml"

dirname = os.path.dirname(os.path.abspath(__file__))
hyper_path = os.path.join(dirname, hyperparams_path)


def hyperparam_handler(route_name: str, chunk_count: str = None):
    
    key = "TALK_TO_FILE_CHITCHAT" if route_name in ["chitchat", "gratitude"] else "TALK_TO_FILE"
    hp = read_yaml_file(hyper_path)[key]

    #token_count = token_counter(chunk_text, model_name="gpt-4")
    
    hp_dict = {
        "model_name": hp["model"],
        "temperature": hp["temperature"],
        "max_tokens": hp["max_tokens"]
    }

    return hp_dict


if __name__ == "__main__":
    instruction = input("Enter the text of the File you want to Chat with ")
    print(hyperparam_handler(instruction))
