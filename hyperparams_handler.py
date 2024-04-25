import os
from utils import read_yaml_file

hyperparams_path = "configs/hyperparams.yaml"

dirname = os.path.dirname(os.path.abspath(__file__))
hyper_path = os.path.join(dirname, hyperparams_path)


def hyperparam_handler(route_name: str):
    key = (
        "TALK_TO_FILE_CHITCHAT_GEMINI"
        if route_name in ["chitchat", "gratitude"]
        else "TALK_TO_FILE_GEMINI"
    )
    hp = read_yaml_file(hyper_path)[key]

    # token_count = token_counter(chunk_text, model_name="gpt-4")

    hp_dict = {
        "temperature": hp["temperature"],
        "max_output_tokens": hp["max_output_tokens"],
        "top_p": hp["top_p"],
        "stop_sequences": hp["stop_sequences"],
    }

    return hp_dict


if __name__ == "__main__":
    route_name = input("Enter Route Name: ")
    print(hyperparam_handler(route_name))
