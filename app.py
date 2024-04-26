import os
import uuid

import base64
import asyncio
import streamlit as st
from dotenv import load_dotenv

from models.T2F import MultiFileQAItems
from process_generation import GenerativeLayer

from utils import docx_to_pdf

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


def clear_chat():
    """
    Clears the chat history by removing all keys from the session state.
    """
    for key in st.session_state.keys():
        del st.session_state[key]


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
    plain_text_files_list = []

    for file_path in file_paths:
        file_dict = {"file_id": str(uuid.uuid4()),
                     "file_name": os.path.basename(file_path),
                     "file_url": file_path
                     }
        print(type(file_dict["file_id"]))
        print(file_dict["file_id"])
        plain_text_files_list.append(file_dict)

    return plain_text_files_list


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


async def talk_to_multiple_files(prompt: MultiFileQAItems):
    answers_func_list = []

    # Concurrently send the question & the whole text of each file to Anthropic
    for file in prompt.plain_text_files_list:
        file_id, file_name, file_url = file.values()

        answers_func_list.append(GenerativeLayer.process_get_anthropic_answer(file_id,
                                                                              file_name,
                                                                              file_url,
                                                                              prompt.question))

    # Gather the answers from Anthropic in a list
    anthropic_answers = await asyncio.gather(*answers_func_list)

    # Convert Anthropic Answers to a single string for use in Final Cognition Layer with GPT-4
    anthropic_answers_str = ''.join(anthropic_answers)

    return await GenerativeLayer.process_ttmf_final_answer(anthropic_answers_str, prompt.question)


async def main():
    if uploaded_pdfs:

        with st.spinner("Uploading and Reading PDF..."):
            file_paths = save_files(uploaded_pdfs)
            plain_text_files_list = pdf_reader(file_paths)

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

            incoming_question = MultiFileQAItems(plain_text_files_list=plain_text_files_list,
                                                 question=user_message)
            print(f'Incoming Question received: {incoming_question}')
            with st.spinner("Generating Response"):
                with st.chat_message("assistant"):
                    gen_response = await talk_to_multiple_files(incoming_question)
                    st.write(gen_response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": gen_response}
                    )
        else:
            clear_chat()


if __name__ == "__main__":
    asyncio.run(main())
