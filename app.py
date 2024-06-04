import os

from openai import OpenAI
import base64
import streamlit as st
from dotenv import load_dotenv

from assistant import create_vector_store, create_thread, delete_vector_store

from utils import read_pdf, docx_to_pdf

load_dotenv()

openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
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
    file_paths = []

    for uploaded_file in uploaded_files:
        uploaded_file_name = uploaded_file.name
        file_path = f"{temp_folder_path}/" + uploaded_file_name

        # with open(file_path, "wb") as temp_file:
        #     temp_file.write(uploaded_file.getvalue())
        # Save the uploaded file to disk
        temp_file_path = os.path.join(temp_folder_path, uploaded_file_name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        if uploaded_file_name.endswith(".docx"):
            file_path = docx_to_pdf(temp_file_path)

        elif uploaded_file_name.endswith(".pdf"):
            print(uploaded_file)
            print(uploaded_file_name)

            # Pass the file path to read_pdf
            file_text = read_pdf(temp_file_path)

            with open(file_path, "wb") as temp_file:
                temp_file.write(file_text.encode('utf-8'))
        else:
            pass

        file_paths.append(file_path)

    print(f"These are the file paths: {file_paths} ")
    # clear_chat()

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
def vector_store_create(_openai_client, file_paths):
    vector_id = create_vector_store(_openai_client, file_paths)

    print(f"I am being run {vector_id}")

    return vector_id


# sidebar
with st.sidebar:
    st.title("🤗💬 Talk to File")
    # Upload PDF file
    uploaded_pdfs = st.file_uploader("Upload your PDF", type=["pdf", "docx"], accept_multiple_files=True)
    if uploaded_pdfs:
        st.markdown(
            "<h3 style= 'text-align:center; color: white;'> PDF Preview </h2>",
            unsafe_allow_html=
            True,
        )
        # print(uploaded_pdfs)
        displayPDF(uploaded_pdfs)


def assistant_response(file_paths, user_question, vector_store_id):
    assistant_id = "asst_fEH4yyo9gArp3vx7DNNxPLZS"

    # Update the assistant with the vector_store_id
    assistant = openai_client.beta.assistants.update(
        assistant_id=assistant_id, tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}}
    )

    thread_id = create_thread(openai_client, user_question)

    run = openai_client.beta.threads.runs.create_and_poll(
        thread_id=thread_id, assistant_id=assistant_id
    )

    messages = list(openai_client.beta.threads.messages.list(thread_id=thread_id, run_id=run.id))

    message_content = messages[0].content[0].text
    annotations = message_content.annotations
    # citations = []
    citations_set = set()
    for index, annotation in enumerate(annotations):
        message_content.value = message_content.value.replace(annotation.text, f"[{index}]")
        if file_citation := getattr(annotation, "file_citation", None):
            cited_file = openai_client.files.retrieve(file_citation.file_id)
            # citations.append(f"[{index}] {cited_file.filename}")
            citations_set.add(f"{cited_file.filename}")

    citations_final = ", ".join(citations_set)
    print(messages)

    # Delete the used thread
    openai_client.beta.threads.delete(thread_id=thread_id)

    return message_content.value, citations_final


def main():
    if uploaded_pdfs:

        with st.spinner("Uploading and Reading PDF..."):

            file_paths = save_files(uploaded_pdfs)
            vector_store_id = vector_store_create(openai_client, file_paths)

        if "vector_store_id" not in st.session_state:
            st.session_state["vector_store_id"] = vector_store_id

        if "messages" not in st.session_state:
            st.session_state.messages = []

        print(st.session_state)

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
                    gen_response, citations = assistant_response(file_paths, user_message, vector_store_id)
                    st.markdown(f'{gen_response} <br> <b>References:</b> <br> {citations}', unsafe_allow_html=True)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f'{gen_response} \n References: {citations}'}
                    )

    else:
        if "vector_store_id" in st.session_state:
            delete_vector_store(openai_client, st.session_state["vector_store_id"])
            print(f'Vector Store deleted')
        clear_chat()


if __name__ == "__main__":
    main()
