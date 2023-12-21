import os
import yaml
import math
import time
import tiktoken
import subprocess
import pinecone
from typing import List
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient, AnalysisFeature
from PyPDF2 import PdfReader, PdfWriter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings


# azure_ocr_endpoint = os.environ.get("AZURE_OCR_ENDPOINT")
# azure_ocr_key = os.environ.get("AZURE_SECRET_KEY")
# pinecone_api_key = os.environ.get("PINECONE_API_KEY")
# pinecone_env = os.environ.get("PINECONE_ENV")
azure_ocr_endpoint = os.environ.get("AZURE_OCR_ENDPOINT")
azure_ocr_key = os.environ.get("AZURE_SECRET_KEY")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_env = os.environ.get("PINECONE_ENV")

pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
index_name = "t2findex"
index = pinecone.GRPCIndex(index_name)


def read_text_file(file_path: str) -> str:
    with open(file_path, "r") as f:
        content = f.read()
    return content


def read_yaml_file(file_path: str):
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        raise ValueError(f"The provided path '{file_path}' is not a valid file.")

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
            file.close()
            return data
    except Exception as e:
        raise RuntimeError(f"Error reading or parsing YAML file: {e}")


def read_pdf(input_file_path: str):
    """Returns the text of the input PDF if the PDF has already been OCR'd

    :param input_file_path: path to PDF
    :type input_file_path: str
    :return: string with PDF text content if PDF has already been OCR'd
    :rtype: str
    """

    reader = PdfReader(input_file_path)
    for i in range(len(reader.pages)):
        extracted_page = reader.pages[i].extract_text()
    
    if len(extracted_page) > 0:
        pages, page_type = data_chunker(input_file_path), "non-ocr"
        return pages, page_type
    else:
        file_text, page_type = azure_ocr(input_file_path), "ocr"
        return file_text, page_type


def token_counter(string: str, model_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(f"{model_name}")
    num_tokens = len(encoding.encode(string))
    return num_tokens


def data_chunker(file_path: str):
    # managing chunk size with recursivecharsplitter
    splitter_page = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=0, length_function=len
    )

    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split(splitter_page)

    return pages


def azure_ocr(input_file_path: str):
    # Set up Azure OCR client

    document_analysis_client = DocumentAnalysisClient(
        endpoint=azure_ocr_endpoint, credential=AzureKeyCredential(azure_ocr_key)
    )

    # Pass input Prior_Art_PDF to Azure OCR client
    with open(input_file_path, "rb") as f:
        poller = document_analysis_client.begin_analyze_document(
            "prebuilt-layout", document=f, features=[AnalysisFeature.LANGUAGES]
        )

    ocr_results = poller.result()

    results_dict = ocr_results.to_dict()
    
    file_text = []
    
    # Loop through each page in results_dict to join the lines per page in a single string
    for i in range(len(results_dict['pages'])):
        elem_dict = {}
        content_list = [results_dict['pages'][i]['lines'][k]['content'] 
                        for k in range(len(results_dict['pages'][i]['lines']))]
        content_str = ' '.join(content_list)
        elem_dict['page_content'] = content_str
        elem_dict['page_number'] = i + 1
        file_text.append(elem_dict)
    
    return file_text


def docx_to_pdf(input_path: str) -> str:
    output_directory = os.path.dirname(
        input_path
    )  # Extracts the directory from the input path
    output_filename = os.path.splitext(os.path.basename(input_path))[0] + ".pdf"
    output_path = os.path.join(output_directory, output_filename)

    try:
        subprocess.run(
            [
                "soffice",
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                output_directory,
                input_path,
            ],
            check=True,
        )
        return output_path  # Return the full path of the output PDF
    except subprocess.CalledProcessError:
        return None

def vectoriser(text: str) -> List[float]:
    embeddings = OpenAIEmbeddings(
        deployment = "right-co-pilot-embedding-ada-002",
        openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
        openai_api_base=os.environ.get("AZURE_OPENAI_API_BASE"),
        openai_api_type=os.environ.get("AZURE_OPENAI_API_TYPE"),
        openai_api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    )
    return embeddings.embed_query(text)


def embeds_metadata(pages,page_type):

    chunk_metadata = []
    chunk_embeds = []

    if page_type == "non-ocr":

        ids = [f"ID-{id}" for id in range(len(pages))]
        for page in pages:
            record_metadata = {
                "chunk_page": page.metadata["page"] + 1,
                "chunk_text": page.page_content,
            }

            chunk_embeds.append(vectoriser(page.page_content))
            chunk_metadata.append(record_metadata)

    else:

        ids = [f"ID-{id}" for id in range(len(pages))]
        for page in pages:
            record_metadata = {
                "chunk_page": page["page_number"] ,
                "chunk_text": page["page_content"], 
            }

            chunk_embeds.append(vectoriser(page["page_content"]))
            chunk_metadata.append(record_metadata)

    return ids, chunk_metadata, chunk_embeds


def indexer(pages, page_type):
    index.delete(delete_all=True)
    ids, chunk_metadata, chunk_embeds = embeds_metadata(pages, page_type)
    file = list(zip(ids, chunk_embeds, chunk_metadata))
    index.upsert(vectors=file)

    
def retreiver(query: str) -> str:
    vector_count = index.describe_index_stats()["total_vector_count"]
    top_k = 5 if vector_count > 5 else vector_count
    query_vector = vectoriser(query)
    query_results = index.query(
        vector=query_vector, top_k=top_k, include_values=False, include_metadata=True
    )

    chunk_final = "\n\n".join(
        f"Chunk Page Number: {query_results['matches'][i]['metadata']['chunk_page']}\n\n"
        f"Chunk Text: {query_results['matches'][i]['metadata']['chunk_text']}\n\n"
        "---- END OF CHUNK ----"
        for i in range(top_k)
    )
    return chunk_final



# query = "give me a summary "

# print(retreiver(query))

# pages = data_chunker("temp_data/NPL- D2 Document (3).pdf")
# indexer(pages)

#print(azure_ocr("temp_data/US2023214776A1 (2).pdf"))

# pages, status= read_pdf("/Users/rohitsaluja/Documents/Github-silo-ai/RightHub/T2F-stlit/temp_data/US2023214776A1 (2).pdf")
# indexer(pages, status)



