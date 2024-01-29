import os
from utils import read_yaml_file
from semantic_router import Route
from semantic_router.layer import RouteLayer
from semantic_router.encoders import OpenAIEncoder


# Read the routes from the YAML file
routes_path = "t2f_routes/talk_to_file_routes.yaml"
dirname = os.path.dirname(os.path.abspath(__file__))
routes_path_ttf = os.path.join(dirname, routes_path)

routes = read_yaml_file(routes_path_ttf)["TALK_TO_FILE_ROUTES"]

# Define the routes
chitchat = Route(name="chitchat", utterances=routes["CHITCHAT"]["UTTERENCES"])
gratitude = Route(name="gratitude", utterances=routes["GRATITUDE"]["UTTERENCES"])
sports_talk = Route(name="sports_talk", utterances=routes["SPORTS_TALK"]["UTTERENCES"])
politics_discussion = Route(name="politics_discussion", utterances=routes["POLITICS_DISCUSSION"]["UTTERENCES"])
chunk_discussions = Route(name="chunk_discussions", utterances=routes["CHUNKS_DISCUSSION"]["UTTERENCES"])
chunk_discussions = Route(name="prompt_leaks", utterances=routes["PROMPT_LEAKS"]["UTTERENCES"])


def get_route_name(question: str) -> str:
    """
    Get the name of the route based on the given question.

    Args:
        question (str): The question to determine the route for.

    Returns:
        str: The name of the route.
    """
    routes = [chitchat, gratitude, sports_talk, politics_discussion, chunk_discussions]
    encoder = OpenAIEncoder(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    rl = RouteLayer(encoder=encoder, routes=routes)
    return rl(question).name

if __name__ == "__main__":
    question = "give me the reference number?"
    route_name = get_route_name(question)
    print(f"The route for the question '{question}' is: {route_name}")

