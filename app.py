import os
import time
import openai
from llama_index.core import SimpleDirectoryReader
from openai import AzureOpenAI, OpenAI
from dotenv import load_dotenv
import base64
import asyncio
import streamlit as st
from starlette.responses import StreamingResponse

from t2f_router import get_route_name
from utils import (token_counter, docx_to_pdf, build_sentence_window_index,
                   get_sentence_window_query_engine, convert_to_llidx_docs)
from prompt_creator import prompt_creator
from hyperparams_handler import hyperparam_handler
from llama_index.llms.openai import OpenAI

load_dotenv()

temp_path = "temp_data"
dirname = os.path.dirname(os.path.abspath(__file__))
temp_folder_path = os.path.join(dirname, temp_path)

st.set_page_config(layout="wide")

col1, col2 = st.columns(spec=[0.6, 0.4], gap="medium")

css = """
<style>
    [data-testid="stSidebar"]{
        
        min-width: 800px;
        max-width: 900px;
    }
</style>
"""
st.markdown(css, unsafe_allow_html=True)

st.markdown(
    """
    <style>
        button[data-testid="baseButton-header"] {
            display: none;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def save_file(uploaded_file):
    uploaded_file_name = uploaded_file.name
    file_path = f"{temp_folder_path}/" + uploaded_file_name
    with open(file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.getvalue())

    if uploaded_file_name.endswith(".docx"):
        file_path = docx_to_pdf(file_path)
    else:
        pass
    return file_path


@st.cache_data
def displayPDF(file_path):
    print("display PDF run")
    # Read local PDF file as bytes:
    with open(file_path, "rb") as file:
        bytes_data = file.read()

    # Convert bytes to base64 for embedding
    base64_pdf = base64.b64encode(bytes_data).decode("utf-8")

    # Create an iframe to display the PDF
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800px" type="application/pdf"></iframe>'

    # Display the PDF
    st.markdown(pdf_display, unsafe_allow_html=True)


# sidebar
with st.sidebar:
    st.title("🤗💬 Talk to File")
    # Upload PDF file
    uploaded_pdf = st.file_uploader("Upload your PDF", type=["pdf"])
    # Define the LLM to use for creating the index
    llm = OpenAI(model="gpt-4-turbo-preview", temperature=0.1)

    if uploaded_pdf is not None:
        saved_file_path = save_file(uploaded_pdf)

        # Check if the current file has been processed already
        if "file_processed" not in st.session_state or st.session_state["file_processed"] != uploaded_pdf.name:
            with st.spinner("Uploading and Reading PDF..."):
                # Convert file to list of LlamaIndex Documents
                docs = convert_to_llidx_docs(saved_file_path)
                # Build the Sentence Window index
                index = build_sentence_window_index(docs, llm=llm)
                # Build the Query Engine
                query_engine = get_sentence_window_query_engine(index)
            # Update the session state to indicate this file has been processed
            st.session_state["file_processed"] = uploaded_pdf.name
            st.session_state["index"] = index
            st.session_state["query_engine"] = query_engine

        st.markdown(
            "<h3 style= 'text-align:center; color: white;'> PDF Preview </h2>",
            unsafe_allow_html=True,
        )
        displayPDF(saved_file_path)
    else:
        st.cache_data.clear()
        # Optionally, clear the processed file flag when no file is uploaded
        if "file_processed" in st.session_state:
            del st.session_state["file_processed"]


def string_streamer(input_string: str):
    """
    Stream a string to the console.

    Args:
        input_string (str): The string to stream.
    """
    stream_str = ""
    for char in input_string:
        stream_str += char
        time.sleep(0.03)
        yield stream_str


def generative_streamer(prompt, hp, chunk_tokens):
    if chunk_tokens < 30000:
        print(chunk_tokens)
        print(hp)

        client = AzureOpenAI(
            azure_endpoint=os.environ.get("AZURE_ENDPOINT"),
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
            api_key=os.environ.get("AZURE_OPENAI_API_KEY")
        )
        model_name = hp["model_name"]

    elif chunk_tokens > 30000 and chunk_tokens < 120000:

        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        model_name = "gpt-4-turbo-preview"

    else:
        st.error("Your file is too big. Supply a smaller file.")

    response = client.chat.completions.create(
        model=model_name,
        messages=prompt,
        temperature=hp["temperature"],
        max_tokens=hp["max_tokens"],
        top_p=hp["top_p"],
        frequency_penalty=hp["frequency_penalty"],
        presence_penalty=hp["presense_penalty"],
        stop=hp["stop_sequences"],
        stream=True,
    )

    full_reply_content = ""

    for chunk in response:
        if len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            full_reply_content += content
            time.sleep(0.02)
            yield full_reply_content


def generative_layer(question: str):
    # hp, prompt = hyperparam_handler(file_text), prompt_creator(file_text, question)
    # df, loaded_embeddings = embeds_metadata_loader(st.session_state["file_uuid"])

    route_name = get_route_name(question)
    print(route_name)

    if route_name == "chitchat" or route_name == "gratitude":
        prompt = prompt_creator(route_name, question)
        hp = hyperparam_handler(route_name)

        streamer = generative_streamer(prompt, hp, chunk_tokens=2000)
        for gen in streamer:
            yield gen

    elif (route_name == "sports_talk" or route_name == "politics_discussion" or
          route_name == "chunk_discussions" or route_name == "prompt_leaks"):
        streamer = string_streamer("My Apologies! I won't be able to answer this question.")
        for stream in streamer:
            yield stream

    else:
        index, query_engine = st.session_state['index'], st.session_state['query_engine']
        response = query_engine.query(question)
        ref_set = set()
        for i in range(len(response.source_nodes)):
            page_number = response.source_nodes[i].node.metadata["page_label"]
            ref_set.add(page_number)
        ref_list = list(ref_set)
        ref_list.sort()

        ref_str = ','.join([str(num) for num in ref_list])

        print(response)
        yield response.response_txt
        ref_streamer = string_streamer(f'<REFERENCES>{ref_str}<END_OF_REFERENCES>')

        for ref in ref_streamer:
            yield ref


def main():
    if uploaded_pdf is not None:

        if "messages" not in st.session_state:
            st.session_state.messages = []

        if user_message := st.chat_input("Ask a question?"):
            st.session_state.messages.append(
                {"role": "user", "content": user_message}
            )

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # If last message is not from assistant, generate a new response
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Generating Response"):
                    message_placeholder = st.empty()
                    gen_iter = generative_layer(user_message)

                    for gen in gen_iter:
                        message_placeholder.markdown(f"{gen}" + "|")
                    message_placeholder.markdown(gen)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": gen}
                    )
    else:
        for key in st.session_state.keys():
            del st.session_state[key]


if __name__ == "__main__":
    main()
