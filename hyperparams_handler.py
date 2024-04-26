import os
from utils import read_yaml_file

hyperparams_path = "configs/hyperparams.yaml"

dirname = os.path.dirname(os.path.abspath(__file__))
hyper_path = os.path.join(dirname, hyperparams_path)


class HyperparamsHandler:
    """Class to handle the hyperparameters of the application."""

    @staticmethod
    def create_hp_dict(hp, hp_key):
        """Create a dictionary of hyperparameters."""
        if hp_key == "GET_ANTHROPIC_ANSWER":
            return {
                "model": hp["model"],
                "temperature": hp["temperature"],
                "max_tokens": hp["max_tokens"],
                "stop_sequences": hp["stop_sequences"]
            }
        else:
            return {
                "model_name": hp["model"],
                "temperature": hp["temperature"],
                "max_tokens": hp["max_tokens"],
                "top_p": hp["top_p"],
                "frequency_penalty": hp["frequency_penalty"],
                "presence_penalty": hp["presence_penalty"],
                "stop_sequences": hp["stop_sequences"],
            }

    @staticmethod
    def _prepare_handle(hp_key: str):
        hp = read_yaml_file(hyper_path)[hp_key]
        return HyperparamsHandler.create_hp_dict(hp, hp_key)

    @staticmethod
    def handle_get_anthropic_answer():
        return HyperparamsHandler._prepare_handle("GET_ANTHROPIC_ANSWER")

    @staticmethod
    def handle_ttmf_final_answer():
        return HyperparamsHandler._prepare_handle("TALK_TO_MULTIPLE_FILES")

# def hyperparam_handler(route_name: str):
#     key = (
#         "TALK_TO_FILE_CHITCHAT_GEMINI"
#         if route_name in ["chitchat", "gratitude"]
#         else "TALK_TO_FILE_GEMINI"
#     )
#     hp = read_yaml_file(hyper_path)[key]
#
#     # token_count = token_counter(chunk_text, model_name="gpt-4")
#
#     hp_dict = {
#         "temperature": hp["temperature"],
#         "max_output_tokens": hp["max_output_tokens"],
#         "top_p": hp["top_p"],
#         "stop_sequences": hp["stop_sequences"],
#     }
#
#     return hp_dict
#
#
# if __name__ == "__main__":
#     route_name = input("Enter Route Name: ")
#     print(hyperparam_handler(route_name))
