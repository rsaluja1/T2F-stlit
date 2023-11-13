import os
from utils import read_yaml_file, token_counter

hyperparams_path = "configs/hyperparams.yaml"

dirname = os.path.dirname(os.path.abspath(__file__))
hyper_path = os.path.join(dirname, hyperparams_path)


def hyperparam_handler(file_text: str):
    token_count = token_counter(file_text, model_name="gpt-4")
    hp = read_yaml_file(hyper_path)["TALK_TO_FILE"]

    hp_dict = {
        "model_name": hp["engine"] if token_count <= 6000 else hp["engine_32k"],
        "temperature": hp["temperature"],
        "max_tokens": hp["max_tokens"],
        "top_p": hp["top_p"],
        "frequency_penalty": hp["frequency_penalty"],
        "presense_penalty": hp["presense_penalty"],
        "stop_sequences": hp["stop_sequences"],
    }

    return hp_dict


if __name__ == "__main__":
    instruction = input("Enter the text of the File you want to Chat with ")
    print(hyperparam_handler(instruction))
