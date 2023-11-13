import os
import time
import openai
import base64
import asyncio
import streamlit as st
from utils import read_pdf, token_counter

from prompt_creator import prompt_creator
from hyperparams_handler import hyperparam_handler


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


def generative_layer(file_text: str, question: str) -> str:
    hp, prompt = hyperparam_handler(file_text), prompt_creator(file_text, question)
    token_count = token_counter(file_text, "gpt-4")

    if token_count <= 30000:
        openai.api_type = "azure"
        openai.api_base = "https://copilot-gpt4.openai.azure.com"
        openai.api_version = "2023-07-01-preview"
        openai.api_key = "e660f3b54a5649858353d641a0bdb0d0"

        response = openai.ChatCompletion.create(
            engine=hp["model_name"],
            messages=prompt,
            temperature=hp["temperature"],
            max_tokens=hp["max_tokens"],
            top_p=hp["top_p"],
            frequency_penalty=hp["frequency_penalty"],
            presence_penalty=hp["presense_penalty"],
            stop=hp["stop_sequences"],
            stream=True,
        )

        collected_messages = []

        for chunk in response:
            chunk_message = (
                chunk["choices"][0]["delta"] if chunk["choices"] else ""
            )  # extract the message

            # if not isinstance(chunk_message, str):
            #     time.sleep(0.04)
            #     yield chunk_message.get("content", "")

            collected_messages.append(chunk_message)

            full_reply_content = "".join(
                [
                    m.get("content", "")
                    for m in collected_messages
                    if not isinstance(m, str)
                ]
            )

            time.sleep(0.02)
            yield full_reply_content

    elif token_count > 30000 and token_count < 120000:
        openai.api_type = "open_ai"
        openai.api_base = "https://api.openai.com/v1"
        openai.api_version = None
        openai.api_key = "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"
        response = openai.ChatCompletion.create(
            model="gpt-4-1106-preview",
            messages=prompt,
            temperature=hp["temperature"],
            max_tokens=hp["max_tokens"],
            top_p=hp["top_p"],
            frequency_penalty=hp["frequency_penalty"],
            presence_penalty=hp["presense_penalty"],
            stop=hp["stop_sequences"],
            stream=True,
        )

        collected_chunks = []
        collected_messages = []

        for chunk in response:
            collected_chunks.append(chunk)  # save the event response
            chunk_message = chunk["choices"][0]["delta"]  # extract the message
            collected_messages.append(chunk_message)  # save the message
            full_reply_content = "".join(
                [m.get("content", "") for m in collected_messages]
            )
            time.sleep(0.05)
            yield full_reply_content

    else:
        st.error("Your file is too big. Supply a smaller file.")


@st.cache_data
def displayPDF(uploaded_file):
    # Read file as bytes:
    bytes_data = uploaded_file.getvalue()
    base64_pdf = base64.b64encode(bytes_data).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800px" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


# sidebar
with st.sidebar:
    st.title("🤗💬 Talk to File")
    # Upload PDF file
    uploaded_pdf = st.file_uploader("Upload your PDF", type="pdf")
    if uploaded_pdf is not None:
        st.markdown(
            "<h3 style= 'text-align:center; color: white;'> PDF Preview </h2>",
            unsafe_allow_html=True,
        )
        displayPDF(uploaded_pdf)

    # add_vertical_space(20)
    # st.write("Made by RightHub AI 🔥")


@st.cache_data
def save_file(uploaded_file):
    file_path = f"{temp_folder_path}/" + uploaded_file.name
    with open(file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.getvalue())
    return file_path


@st.cache_data
def pdf_reader(file_path):
    pdf_text = read_pdf(file_path)
    return pdf_text


def main():
    if uploaded_pdf is not None:
        with st.spinner("Uploading and Reading PDF..."):
            file_path = save_file(uploaded_pdf)
            pdf_text = pdf_reader(file_path)
        token_count = token_counter(pdf_text, "gpt-4")
        if token_count >= 120000:
            st.error("Your file is too big. Please upload a smaller file.")
        else:
            if "messages" not in st.session_state:
                st.session_state.messages = []

            if st.session_state.messages:
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

            if user_message := st.chat_input("Ask a question?"):
                st.session_state.messages.append(
                    {"role": "user", "content": user_message}
                )
                with st.chat_message("user"):
                    st.markdown(user_message)

                with st.spinner("Generating Response"):
                    with st.chat_message("assistant"):
                        message_placeholder = st.empty()
                        gen_iter = generative_layer(pdf_text, user_message)

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
    # os.system(f"streamlit run /Users/rohitsaluja/Documents/Github-silo-ai/RightHub/talk-to-file/app/streamlit/demo.py")
    main()
