import os
import time
from openai import AzureOpenAI


client = AzureOpenAI(
    azure_endpoint=os.environ.get("AZURE_ENDPOINT"),
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
    api_key=os.environ.get("AZURE_OPENAI_API_KEY")
    )

response = client.chat.completions.create(
        model="right-co-pilot-gpt4-8k",
        messages=[{"role": "user","content": "what is capital of paris?"}],
        stream=True
)

#res_obj = response

# collected_messages = []
# for chunk in response:
#     chunk_message = (
#         chunk["choices"][0]["delta"] if chunk["choices"] else "")


# for chunk in response:
#     print(len(chunk.choices))

collected_messages = []
full_reply_content = ""

for chunk in response:
    if len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
        content = chunk.choices[0].delta.content
        full_reply_content += content
        os.system('clear')
        print(full_reply_content)
        time.sleep(0.02)

def string_streamer(input_string: str):
    """
    Stream a string to the console.

    Args:
        input_string (str): The string to stream.
        delay (float, optional): The delay between each character. Defaults to 0.02.
    """
    stream_str = ""
    for char in input_string:
        stream_str += char
        yield stream_str


stream_obj = string_streamer("Who are you?")

for str in stream_obj:
    os.system('clear')
    print(str)
    time.sleep(0.02)
