import os
import yaml
import math
import time
import uuid
import torch
import tiktoken
import subprocess
import pinecone
import pandas as pd
from typing import List
import torch.nn.functional as F
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient, AnalysisFeature
from PyPDF2 import PdfReader, PdfWriter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings


dirname = os.path.dirname(os.path.abspath(__file__))
data_dir_path = os.path.join(dirname, "data")



# azure_ocr_endpoint = os.environ.get("AZURE_OCR_ENDPOINT")
# azure_ocr_key = os.environ.get("AZURE_SECRET_KEY")
# pinecone_api_key = os.environ.get("PINECONE_API_KEY")
# pinecone_env = os.environ.get("PINECONE_ENV")
azure_ocr_endpoint = os.environ.get("AZURE_OCR_ENDPOINT")
azure_ocr_key = os.environ.get("AZURE_SECRET_KEY")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_env = os.environ.get("PINECONE_ENV")

import os
import yaml
import tiktoken
import subprocess
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient, AnalysisFeature
from PyPDF2 import PdfReader, PdfWriter


azure_ocr_endpoint = os.environ.get("AZURE_OCR_ENDPOINT")
azure_ocr_key = os.environ.get("AZURE_SECRET_KEY")


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

def data_chunker(file_path: str):
    """
    Splits the content of a PDF file into chunks of text.

    This function uses a RecursiveCharacterTextSplitter to divide the text from a PDF file,
    specified by file_path, into chunks. Each chunk is up to 2000 characters, with no overlap
    between chunks.

    Parameters:
    file_path (str): Path to the PDF file to be split into chunks.

    Returns:
    A list of text chunks from the PDF file.
    """
    # managing chunk size with recursivecharsplitter
    splitter_page = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=0, length_function=len
    )

    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split(splitter_page)

    return pages


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
        pages, ocr_status = data_chunker(input_file_path), "non-ocr"
        return pages, ocr_status
    else:
        file_text, ocr_status = azure_ocr(input_file_path), "ocr"
        return file_text, ocr_status


def token_counter(string: str, model_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(f"{model_name}")
    num_tokens = len(encoding.encode(string))
    return num_tokens



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
    """
    Converts a text string into a list of numerical embeddings using OpenAI's Embeddings API.

    Parameters:
    text (str): Input text to be vectorized.

    Returns:
    List[float]: Numerical embeddings of the input text.
    """
    embeddings = OpenAIEmbeddings(
        deployment = "right-co-pilot-embedding-ada-002",
        openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
        openai_api_base=os.environ.get("AZURE_OPENAI_API_BASE"),
        openai_api_type=os.environ.get("AZURE_OPENAI_API_TYPE"),
        openai_api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    )
    return embeddings.embed_query(text)


def embeds_metadata(pages,page_type):

    chunk_texts = []
    chunk_page = []
    chunk_embeds= []

    for page in pages:
        chunk_texts.append(page.page_content) if page_type == "non-ocr" else chunk_texts.append(page["page_content"])
        chunk_page.append(page.metadata["page"]+1) if page_type == "non-ocr" else chunk_page.append(page["page_number"])
        chunk_embeds.append(vectoriser(page.page_content)) if page_type == "non-ocr" else chunk_embeds.append(vectoriser(page.page_content))
            

    return chunk_texts,chunk_page,chunk_embeds

    
def save_embeds_metadata(pages,page_type):
   
   file_uuid = uuid.uuid4()

   chunk_texts,chunk_page,chunk_embeds = embeds_metadata(pages, page_type)
    
   os.makedirs(data_dir_path, exist_ok=True)
   torch.save(torch.tensor(chunk_embeds), f'{data_dir_path}/{file_uuid}-pt.pt')
   df = pd.DataFrame({'Chunk_Text': chunk_texts, 'Page_Number': chunk_page})
   df.to_csv(f'{data_dir_path}/{file_uuid}-df.csv', header=True, index = False)

   return file_uuid


def embeds_metadata_loader(file_uuid):
    df= pd.read_csv(f'{data_dir_path}/{file_uuid}-df.csv', header=0)
    loaded_embeddings = torch.load(f'{data_dir_path}/{file_uuid}-pt.pt')

    return df, loaded_embeddings

def retreiver(query: str, loaded_embeddings: torch.Tensor, chunks_df: pd.DataFrame ) -> str:
    """
    Retrieves and formats the top 5 chunks from a DataFrame based on cosine similarity to a query embedding.

    Parameters:
    query (str): The query string.
    loaded_embeddings (torch.Tensor): Tensor of t2f-document embeddings.
    chunks_df (pd.DataFrame): DataFrame containing chunk details.

    Returns:
    str: Formatted string of the top 5 chunks with their page numbers and texts.
    """


    query_embedding = torch.tensor(vectoriser(query)).view(1, -1)
    cosine_similarity_scores = F.cosine_similarity(query_embedding, loaded_embeddings)
    top_k = torch.topk(cosine_similarity_scores, k=5)
    top_chunks = top_k.indices.tolist()
    retrieved_chunks_df = chunks_df.iloc[top_chunks]
    chunk_final = "\n\n".join(
    f"Chunk Page Number: {row['Page_Number']}\n\n"
    f"Chunk Text: {row['Chunk_Text']}\n\n"
    "---- END OF CHUNK ----"
    for index, row in retrieved_chunks_df.iterrows()
)
    return chunk_final


if __name__ == "__main__":
    
    pages, ocr_status= read_pdf("/Users/rohitsaluja/Documents/Github-silo-ai/RightHub/T2F-stlit/temp_data/NPL- D2 Document.pdf")
    file_uuid = save_embeds_metadata(pages,ocr_status)
    df, loaded_embeddings = embeds_metadata_loader(file_uuid)


    query = "Who was the author supervised by?"

    retreived_chunks = retreiver(query,loaded_embeddings,df)
    print(retreived_chunks)

    # pages = data_chunker("temp_data/managing_ai_risks.pdf")
    # hunk_texts,chunk_page,chunk_embeds = embeds_metadata(pages)
    # save_embeds_metadata()


    # indexer(pages)

    #print(azure_ocr("temp_data/US2023214776A1 (2).pdf"))

    # pages, status= read_pdf("/Users/rohitsaluja/Documents/Github-silo-ai/RightHub/T2F-stlit/temp_data/US2023214776A1 (2).pdf")
    # indexer(pages, status)



