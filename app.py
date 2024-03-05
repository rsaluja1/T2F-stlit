import os
import time
import openai
from groq import Groq
from openai import AzureOpenAI, OpenAI
import base64
import asyncio
import streamlit as st
from t2f_router import get_route_name
from utils import azure_ocr_to_vectorize, token_counter, docx_to_pdf, data_chunker,retreiver,save_embeds_metadata,embeds_metadata_loader
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


# def displayPDF(uploaded_file):
#     # Read file as bytes:
#     bytes_data = uploaded_file.getvalue()
#     st.write(uploaded_file.getvalue())
#     base64_pdf = base64.b64encode(bytes_data).decode("utf-8")
#     pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800px" type="application/pdf"></iframe>'
#     st.markdown(pdf_display, unsafe_allow_html=True)


@st.cache_data
def embeds_metadata_save(file_path):
    print(file_path)
    pages = azure_ocr_to_vectorize(file_path)
    print("saving embeds metada")
    file_uuid = save_embeds_metadata(pages)
    return file_uuid


# @st.cache_data
# def chunker_indexer(file_path):
#     print("chunker running")
#     pages, ocr_status = read_pdf(file_path)
#     #indexer(pages)
#     indexer(pages, ocr_status)

# sidebar
# with st.sidebar:
#     st.title("🤗💬 Talk to File")
#     # Upload PDF file
#     uploaded_pdf = st.file_uploader("Upload your PDF", type=["pdf", "docx"])
#     if uploaded_pdf is not None:
#         saved_file_path = save_file(uploaded_pdf)
#         st.markdown(
#             "<h3 style= 'text-align:center; color: white;'> PDF Preview </h2>",
#             unsafe_allow_html=True,
#         )
#         with st.spinner("Uploading and Reading PDF..."):
#             chunker_indexer(saved_file_path)
#             displayPDF(saved_file_path)
#     else:
#         st.cache_data.clear()

#     # add_vertical_space(20)
#     # st.write("Made by RightHub AI 🔥")
    
# sidebar
with st.sidebar:
    st.title("🤗💬 Talk to File")
    # Upload PDF file
    uploaded_pdf = st.file_uploader("Upload your PDF", type=["pdf"])

    if uploaded_pdf is not None:
        saved_file_path = save_file(uploaded_pdf)
        # Check if the current file has been processed already
        if "file_processed" not in st.session_state or st.session_state["file_processed"] != uploaded_pdf.name:
           
            with st.spinner("Uploading and Reading PDF..."):
                file_uuid = embeds_metadata_save(saved_file_path)
            # Update the session state to indicate this file has been processed
            st.session_state["file_processed"] = uploaded_pdf.name
            st.session_state["file_uuid"] = file_uuid
            st.session_state["df"], st.session_state["loaded_embeddings"] = embeds_metadata_loader(st.session_state["file_uuid"])

        
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
        if "file_uuid" in st.session_state:
            del st.session_state["file_uuid"]

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

def generative_streamer(prompt,hp,chunk_tokens):

    if chunk_tokens <30000:
        print(chunk_tokens)
        print(hp)
    
        client = Groq(
        api_key="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",
        )
        model_name = "mixtral-8x7b-32768"

    else:
        st.error("Your file is too big. Supply a smaller file.")


    stream = client.chat.completions.create(
        model=model_name,
        messages=prompt,
        temperature=hp["temperature"],
        max_tokens=hp["max_tokens"],
        top_p=hp["top_p"],
        stop=hp["stop_sequences"],
        stream=True,
    )


    full_reply_content = ""

    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            full_reply_content += content
            yield full_reply_content


def generative_layer(question: str):
    #hp, prompt = hyperparam_handler(file_text), prompt_creator(file_text, question)
    #df, loaded_embeddings = embeds_metadata_loader(st.session_state["file_uuid"])

    route_name = get_route_name(question)
    print(route_name)

    if route_name == "chitchat" or route_name == "gratitude":
        prompt = prompt_creator(route_name,question)
        hp = hyperparam_handler(route_name)

        streamer = generative_streamer(prompt,hp,chunk_tokens=2000)
        for gen in streamer:
            yield gen

    elif route_name == "sports_talk" or route_name == "politics_discussion" or route_name == "chunk_discussions" or route_name == "prompt_leaks":
        streamer = string_streamer("My Appologies! I won't be able to answer this question.")
        for str in streamer:
            yield str

    else:

        df, loaded_embeddings = st.session_state["df"], st.session_state["loaded_embeddings"]
        route_name = get_route_name(question)
        retreived_chunks = retreiver(question,loaded_embeddings,df)


        prompt, chunk_tokens = prompt_creator(route_name,question,retreived_chunks)
        hp = hyperparam_handler(route_name,chunk_tokens)
        #token_count = token_counter(file_text, "gpt-4")

        streamer = generative_streamer(prompt,hp,chunk_tokens)
        for gen in streamer:
            yield gen

    # #if chunk_tokens <= 30000:
    # client = AzureOpenAI(
    # azure_endpoint=os.environ.get("AZURE_ENDPOINT"),
    # api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
    # api_key=os.environ.get("AZURE_OPENAI_API_KEY")
    # )

    #         # #openai.api_type = st.secrets["AZURE_OPENAI_API_TYPE"]
    #         # openai.api_type = os.environ.get("AZURE_OPENAI_API_TYPE")
    #         # #openai.api_base = st.secrets["AZURE_OPENAI_API_BASE"]
    #         # openai.api_base = os.environ.get("AZURE_OPENAI_API_BASE")
    #         # #openai.api_version = st.secrets["AZURE_OPENAI_API_VERSION"]
    #         # openai.api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
    #         # #openai.api_key = st.secrets["AZURE_OPENAI_API_KEY"]
    #         # openai.api_key = os.environ.get("AZURE_OPENAI_API_KEY")

    # response = client.chat.completions.create(
    #     model=hp["model_name"],
    #     messages=prompt,
    #     temperature=hp["temperature"],
    #     max_tokens=hp["max_tokens"],
    #     top_p=hp["top_p"],
    #     frequency_penalty=hp["frequency_penalty"],
    #     presence_penalty=hp["presense_penalty"],
    #     stop=hp["stop_sequences"],
    #     stream=True,
    #     )

    # full_reply_content = ""

    # for chunk in response:
    #     if len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
    #         content = chunk.choices[0].delta.content
    #         full_reply_content += content
    #         time.sleep(0.03)
    #         yield full_reply_content
                


        # collected_messages = []

        # for chunk in response:
        #     chunk_message = (
        #         chunk["choices"][0]["delta"] if chunk["choices"] else ""
        #     )  # extract the message

        #     # if not isinstance(chunk_message, str):
        #     #     time.sleep(0.04)
        #     #     yield chunk_message.get("content", "")

        #     collected_messages.append(chunk_message)

        #     full_reply_content = "".join(
        #         [
        #             m.get("content", "")
        #             for m in collected_messages
        #             if not isinstance(m, str)
        #         ]
        #     )

        #     time.sleep(0.02)
        #     yield full_reply_content

    #elif chunk_tokens > 30000 and chunk_tokens < 120000:
    # if chunk_tokens <= 120000: 
        
    #     #print(os.environ.get("OPENAI_API_TYPE"), os.environ.get("OPENAI_API_BASE"),os.environ.get("OPENAI_API_KEY"))
    #     #openai.api_type = st.secrets["OPENAI_API_TYPE"]
    #     openai.api_type = os.environ.get("OPENAI_API_TYPE")
    #     #openai.api_base = st.secrets["OPENAI_API_BASE"]
    #     openai.api_base = os.environ.get("OPENAI_API_BASE")
    #     openai.api_version = None
    #     #openai.api_key = st.secrets["OPENAI_API_KEY"]
    #     openai.api_key = os.environ.get("OPENAI_API_KEY")
        
    #     response = openai.ChatCompletion.create(
    #         model="gpt-4-1106-preview",
    #         messages=prompt,
    #         temperature=hp["temperature"],
    #         max_tokens=hp["max_tokens"],
    #         top_p=hp["top_p"],
    #         frequency_penalty=hp["frequency_penalty"],
    #         presence_penalty=hp["presense_penalty"],
    #         stop=hp["stop_sequences"],
    #         stream=True,
    #     )

    #     collected_chunks = []
    #     collected_messages = []

    #     for chunk in response:
    #         collected_chunks.append(chunk)  # save the event response
    #         chunk_message = chunk["choices"][0]["delta"]  # extract the message
    #         collected_messages.append(chunk_message)  # save the message
    #         full_reply_content = "".join(
    #             [m.get("content", "") for m in collected_messages]
    #         )
    #         time.sleep(0.05)
    #         yield full_reply_content

    # else:
    #     st.error("Your file is too big. Supply a smaller file.")


# @st.cache_data
# def pdf_reader(file_path):
#     pdf_text = read_pdf(file_path)
#     return pdf_text


def main():
    if uploaded_pdf is not None:

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

            


        #token_count = token_counter(pdf_text, "gpt-4")
        #if token_count >= 120000:
            #st.error("Your file is too big. Please upload a smaller file.")
        #else:
            

            

            # if user_message := st.chat_input("Ask a question?"):
            #     st.session_state.messages.append(
            #         {"role": "user", "content": user_message}
            #     )
            #     with st.chat_message("user"):
            #         st.markdown(user_message)

            #     with st.spinner("Generating Response"):
            #         with st.chat_message("assistant"):
            #             message_placeholder = st.empty()
            #             gen_iter = generative_layer(pdf_text, user_message)

            #             for gen in gen_iter:
            #                 message_placeholder.markdown(f"{gen}" + "|")
            #             message_placeholder.markdown(gen)
            #             st.session_state.messages.append(
            #                 {"role": "assistant", "content": gen}
            #             )
    
if __name__ == "__main__":
    main()
