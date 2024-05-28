import os
import re
import time
import uuid

import openai
import base64
import asyncio
import vertexai
import streamlit as st
from dotenv import load_dotenv
from typing import List
from hyperparams_handler import hyperparam_handler
from prompt_creator import prompt_creator
from t2f_router import get_route_name

from vertexai.generative_models import GenerativeModel, GenerationConfig, Content
from utils import read_pdf, docx_to_pdf, token_counter_gemini

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


# @st.cache_data
# def save_file(uploaded_file):
#     uploaded_file_name = uploaded_file.name
#     file_path = f"{temp_folder_path}/" + uploaded_file_name
#     with open(file_path, "wb") as temp_file:
#         temp_file.write(uploaded_file.getvalue())

#     if uploaded_file_name.endswith(".docx"):
#         file_path = docx_to_pdf(file_path)
#     else:
#         pass
#     print(f"This is the file path: {file_path} ")
#     return file_path
# from PyPDF2 import PdfFileMerger


def clear_chat():
    """
    Clears the chat history by removing all keys from the session state.
    """
    for key in st.session_state.keys():
        del st.session_state[key]

def query_postprocessing(query: str, file_names: List[str]) -> str:

    query_clean = re.sub(r'\?', '', query)
    query_post = query_clean + (f" in {file_names}?" if "?" in query  else f" in {file_names}")

    return query_post

def extract_file_names(uploaded_files) -> List[str]:

    uploaded_file_names = []
    for uploaded_file in uploaded_files:
        uploaded_file_name = uploaded_file.name
        uploaded_file_names.append(uploaded_file_name)
    
    return uploaded_file_names



@st.cache_data
def save_files(uploaded_files):
    uploaded_file_names = []
    file_paths = []

    for uploaded_file in uploaded_files:
        uploaded_file_name = uploaded_file.name
        file_path = f"{temp_folder_path}/" + uploaded_file_name
        uploaded_file_names.append(uploaded_file_name)

        with open(file_path, "wb") as temp_file:
            temp_file.write(uploaded_file.getvalue())

        if uploaded_file_name.endswith(".docx"):
            file_path = docx_to_pdf(file_path)
        else:
            pass

        file_paths.append(file_path)

    print(f"These are the file paths: {file_paths} ")
    clear_chat()

    return file_paths


# @st.cache_data
# def save_files(uploaded_files):
#     merger = PdfMerger()
#     uploaded_file_names = ""

#     for uploaded_file in uploaded_files:
#         uploaded_file_name = uploaded_file.name
#         file_path = f"{temp_folder_path}/" + uploaded_file_name
#         uploaded_file_names+= uploaded_file_name.replace(".pdf","")

#         with open(file_path, "wb") as temp_file:
#             temp_file.write(uploaded_file.getvalue())

#         if uploaded_file_name.endswith(".docx"):
#             file_path = docx_to_pdf(file_path)
#         else:
#             pass

#         merger.append(file_path)

#     merged_file_path = f"{temp_folder_path}/{uploaded_file_names}-merged.pdf"
#     merger.write(merged_file_path)
#     merger.close()

#     print(f"This is the file path: {merged_file_path} ")
#     return merged_file_path


# display come back
@st.cache_data
def displayPDF(uploaded_files):
    for uploaded_file in uploaded_files:
        # Read file as bytes:
        bytes_data = uploaded_file.getvalue()

        base64_pdf = base64.b64encode(bytes_data).decode("utf-8")
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800px" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)


@st.cache_data
def pdf_reader(file_paths):
    text_prep = ""
    files_text = ""

    for file_path in file_paths:
        #filename = os.path.basename(file_path).replace(".pdf", "")
        filename = os.path.basename(file_path)
        file_text = read_pdf(file_path)

        # text_prep = f"FILE NAME: {filename}\nFILE TEXT: {pdf_text}\n----END OF FILE----\n\n"

        text_prep = "\n".join([
            f"FILE NAME: {filename}",
            f"FILE ID: {str(uuid.uuid4())}",
            f"FILE TEXT: {file_text}",
            "----END OF FILE----",
            ""
        ])

        files_text += text_prep

    return files_text


# sidebar
with st.sidebar:
    st.title("🤗💬 Talk to File")
    # Upload PDF file
    uploaded_pdfs = st.file_uploader("Upload your PDF", type=["pdf"], accept_multiple_files=True)
    if uploaded_pdfs:
        st.markdown(
            "<h3 style= 'text-align:center; color: white;'> PDF Preview </h2>",
            unsafe_allow_html=
            True,
        )
        # print(uploaded_pdfs)
        displayPDF(uploaded_pdfs)


def gemini_response(system_prompt: str, prompt_messages: Content, generation_config: GenerationConfig, question: str):
    vertexai.init(project="dev-envc52ce870", location="europe-north1")

    # load the model
    model = GenerativeModel(
        "gemini-1.5-pro-preview-0409",
        system_instruction=[system_prompt],
        generation_config=generation_config
    )
    # print(prompt_messages)

    contents = prompt_messages
    contents = [Content.from_dict(x) for x in contents]

    user_input = question
    chat = model.start_chat(history=contents)

    response = chat.send_message(
        user_input,
        stream=False
    )

    return response.text


def generative_layer(file_text: str, question: str):
    # token_count = token_counter_gemini(file_text)
    ##question post-processing

    route_name = get_route_name(question)
    print(route_name)

    generation_config = hyperparam_handler(route_name)
    prompt_system, prompt_messages = prompt_creator(route_name, file_text)
    response = gemini_response(prompt_system, prompt_messages, generation_config, question)

    return response


def main():
    if uploaded_pdfs:

        with st.spinner("Uploading and Reading PDF..."):
            file_names = extract_file_names(uploaded_pdfs)
            print(file_names)
            file_paths = save_files(uploaded_pdfs)
            pdf_text = pdf_reader(file_paths)

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
                    user_message_post = query_postprocessing(user_message, file_names)
                    gen_response = generative_layer(pdf_text, user_message_post)
                    st.write(gen_response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": gen_response}
                    )
        else:
            clear_chat()


if __name__ == "__main__":
    main()
