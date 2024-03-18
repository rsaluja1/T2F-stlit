import os
from dotenv import load_dotenv
import yaml
import uuid
import torch
import tiktoken
import subprocess
import pandas as pd
from typing import List
from openai import OpenAI
import torch.nn.functional as F
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient, AnalysisFeature
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings

load_dotenv()
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
from PyPDF2 import PdfReader


def read_text_file(file_path: str) -> str:
    """
    Reads the entire content of a text file and returns it as a string.

    Args:
    file_path (str): The path to the text file to be read.

    Returns:
    str: The content of the file as a string.
    """
    with open(file_path, "r") as f:
        content = f.read()
    return content


def read_yaml_file(file_path: str):
    """
    Reads and parses a YAML file, returning its content as a Python object.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        Any: Parsed content of the YAML file.

    Raises:
        ValueError: If `file_path` is not a valid file path.
        RuntimeError: If reading or parsing the YAML file fails.

    Example:
        data = read_yaml_file("path/to/file.yaml")
    """
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
    Splits the text content of a PDF file into chunks of specified size.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        list: A list of text chunks, where each chunk represents a page of the PDF file.
    """
    # managing chunk size with recursivecharsplitter
    splitter_page = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=0, length_function=len
    )

    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split(splitter_page)

    return pages

def data_chunker_azure(page_text: List[str], metadata: List[dict]):
    """
    Splits the content of a PDF file that has been OCR'd with Azure OCR into chunks of text.

    Each chunk is up to 2000 characters, with no overlap between chunks.

    Parameters:
    page_text (List[str]): A list where each element is a string of the text of each page
    metadata (List[dict]): A list where each element is a dictionary of the page numbers
    corresponding to the strings in page_text

    Returns:
    A list of text chunks from a PDF file that has been OCR'd with Azure OCR.
    """

    # managing chunk size with recursivecharsplitter
    splitter_page = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=0, length_function=len
    )

    pages = splitter_page.create_documents(page_text, metadata)

    return pages



def check_ocr_to_vectorize(input_file_path: str):
    """
    Checks if a PDF needs OCR and vectorizes its content.

    This function reads a PDF file, extracts text from each page, and checks if any page has less than 100 characters. 
    If so, it assumes the PDF needs OCR and uses `data_chunker_azure` to vectorize its content. 
    Otherwise, it uses `data_chunker` to vectorize the content.

    :param input_file_path: The path to the PDF file.
    :type input_file_path: str
    :return: Vectorized content of the PDF.
    """

    reader = PdfReader(input_file_path)

    extract_pages = []
    extracted_page = None
    for i in range(len(reader.pages)):
        extracted_page = reader.pages[i].extract_text()
        extract_pages.append(extracted_page)

    #sometimes even non-ocrd pages have some text in the header that is added later and ocrable. this could give false poisive while chekcing of OCR.
    # we set char count limit to 100 to guard against that false positive. 
    less_than_100_chars = any(len(element) < 100 for element in extract_pages)

    pages = data_chunker(input_file_path) if not less_than_100_chars else azure_ocr_to_vectorize(input_file_path)
        
    return pages


# def read_pdf(input_file_path: str):
#     """
#     Extracts text from a PDF file. 

#     If any page has less than 100 characters, it uses Azure's OCR service to extract the text. 
#     Otherwise, it chunks the text content of the PDF and returns the chunks.

#     :param input_file_path: The path to the PDF file.
#     :type input_file_path: str
#     :return: A tuple of the text content and a string indicating the OCR status ("ocr" or "non-ocr").
#     :rtype: tuple
#     """
#     reader = PdfReader(input_file_path)
#     extract_pages = []

#     for i in range(len(reader.pages)):
#         extracted_page = reader.pages[i].extract_text()
#         extract_pages.append(extracted_page)
#     #sometimes even non-ocrd pages have some text in the header that is added later and ocrable. this could give false poisive while chekcing of OCR.
#     # we set char count limit to 100 to guard against that false positive. 
#     less_than_100_chars = any(len(element) < 100 for element in extract_pages)

#     if less_than_100_chars:
#         file_text, ocr_status = azure_ocr(input_file_path), "ocr"
#         return file_text, ocr_status
#     else:
#         pages, ocr_status = data_chunker(input_file_path), "non-ocr"
#         return pages, ocr_status
    

    # if len(extracted_page) > 0:
    #     pages, ocr_status = data_chunker(input_file_path), "non-ocr"
    #     return pages, ocr_status
    # else:
    #     file_text, ocr_status = azure_ocr(input_file_path), "ocr"
    #     return file_text, ocr_status

    

def token_counter(string: str, model_name: str) -> int:
    """
    Counts the number of tokens in a given string using a specified model's encoding.

    Args:
        string (str): The text to be tokenized.
        model_name (str): The name of the model to use for encoding.

    Returns:
        int: The count of tokens in the string.
    """
    encoding = tiktoken.encoding_for_model(f"{model_name}")
    num_tokens = len(encoding.encode(string))
    return num_tokens


def azure_ocr_to_vectorize(input_file_path: str):
    # Set up Azure OCR client
    document_analysis_client = DocumentAnalysisClient(
        endpoint=azure_ocr_endpoint, credential=AzureKeyCredential(azure_ocr_key)
    )

    # Pass input file to Azure OCR client
    with open(input_file_path, "rb") as f:
        poller = document_analysis_client.begin_analyze_document(
            "prebuilt-layout", document=f, features=[AnalysisFeature.LANGUAGES]
        )

    ocr_results = poller.result()

    results_dict = ocr_results.to_dict()

    metadata = []
    page_text = []

    # Loop through each page in results_dict to join the lines per page in a single string
    for i in range(len(results_dict["pages"])):
        content_list = [
            results_dict["pages"][i]["lines"][k]["content"]
            for k in range(len(results_dict["pages"][i]["lines"]))
        ]
        page_text.append(" ".join(content_list))
        metadata.append({"page": i})

    pages = data_chunker_azure(page_text, metadata)

    return pages



def azure_ocr(input_file_path: str):
    """
    Performs OCR on a document using Azure's Document Analysis Client and returns text by page.

    Args:
        input_file_path (str): File path of the document for OCR.

    Returns:
        list of dict: Each dictionary contains 'page_content' (text of the page) and
                      'page_number', corresponding to each page in the document.

    Note:
        Requires Azure OCR credentials (endpoint and key) for use.
    """
    # Set up Azure OCR client
    document_analysis_client = DocumentAnalysisClient(
        endpoint=azure_ocr_endpoint, credential=AzureKeyCredential(azure_ocr_key)
    )

    # Pass input file to Azure OCR client
    with open(input_file_path, "rb") as f:
        poller = document_analysis_client.begin_analyze_document(
            "prebuilt-layout", document=f, features=[AnalysisFeature.LANGUAGES]
        )

    ocr_results = poller.result()

    results_dict = ocr_results.to_dict()

    file_text = []

    # Loop through each page in results_dict to join the lines per page in a single string
    for i in range(len(results_dict["pages"])):
        elem_dict = {}
        content_list = [
            results_dict["pages"][i]["lines"][k]["content"]
            for k in range(len(results_dict["pages"][i]["lines"]))
        ]
        content_str = " ".join(content_list)
        elem_dict["page_content"] = content_str
        elem_dict["page_number"] = i + 1
        file_text.append(elem_dict)

    return file_text


def docx_to_pdf(input_path: str) -> str:
    """
    Converts a DOCX file to a PDF file using LibreOffice's command-line interface.

    The function converts a DOCX file specified by 'input_path' to a PDF file,
    saving it in the same directory with the same base filename.

    Args:
        input_path (str): The file path of the DOCX file to convert.

    Returns:
        str: The file path of the converted PDF file. Returns None if conversion fails.

    Note:
        LibreOffice must be installed and accessible via the command line as 'soffice'.
    """
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

    client = OpenAI()

    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )

    #embeddings = OpenAIEmbeddings(model="text-embedding-3-small",
                                  #api_key=os.environ.get("OPENAI_API_KEY"))

    
    return response.data[0].embedding

def embeds_metadata(pages):
    chunk_texts = []
    chunk_page = []
    chunk_embeds = []

    for page in pages:
        if len(page.page_content) > 250:
            chunk_texts.append(page.page_content)
            chunk_page.append(page.metadata["page"] + 1)
            chunk_embeds.append(vectoriser(page.page_content))
    
    return chunk_texts, chunk_page, chunk_embeds


# def embeds_metadata(pages, page_type):
#     chunk_texts = []
#     chunk_page = []
#     chunk_embeds = []


#     for page in pages:
#         chunk_texts.append(
#             page.page_content
#         ) if page_type == "non-ocr" else chunk_texts.append(page["page_content"])
#         chunk_page.append(
#             page.metadata["page"] + 1
#         ) if page_type == "non-ocr" else chunk_page.append(page["page_number"])
#         chunk_embeds.append(
#             vectoriser(page.page_content)
#         ) if page_type == "non-ocr" else chunk_embeds.append(
#             vectoriser(page["page_content"])
#         )

#     return chunk_texts, chunk_page, chunk_embeds


def save_embeds_metadata(pages):
    """
    Saves text, page numbers, and their embeddings to files, and returns a unique file UUID.

    Args:
        pages (list): List of page objects.
        page_type (str): Type of pages ('ocr' or 'non-ocr').

    Returns:
        uuid.UUID: The UUID associated with the saved files.

    Note:
        Saves embeddings as a PyTorch tensor and metadata as a CSV in 'data_dir_path'.
    """

    file_uuid = uuid.uuid4()

    chunk_texts, chunk_page, chunk_embeds = embeds_metadata(pages)

    os.makedirs(data_dir_path, exist_ok=True)
    torch.save(torch.tensor(chunk_embeds), f"{data_dir_path}/{file_uuid}-pt.pt")
    df = pd.DataFrame({"Chunk_Text": chunk_texts, "Page_Number": chunk_page})
    df.to_csv(f"{data_dir_path}/{file_uuid}-df.csv", header=True, index=False)

    return file_uuid


def embeds_metadata_loader(file_uuid):
    """
    Loads and returns embeddings and metadata from files using a specified UUID.

    Args:
        file_uuid (uuid.UUID or str): UUID identifying the files to load.

    Returns:
        tuple: (DataFrame with page texts and numbers, tensor of embeddings).

    Note:
        Assumes 'data_dir_path' as the directory for file storage.
    """
    df = pd.read_csv(f"{data_dir_path}/{file_uuid}-df.csv", header=0)
    loaded_embeddings = torch.load(f"{data_dir_path}/{file_uuid}-pt.pt")

    return df, loaded_embeddings


def retreiver(
    query: str, loaded_embeddings: torch.Tensor, chunks_df: pd.DataFrame
) -> str:
    """
    Retrieves and formats the top 5 chunks from a DataFrame based on cosine similarity to a query embedding.

    Parameters:
    query (str): The query string.
    loaded_embeddings (torch.Tensor): Tensor of t2f-document embeddings.
    chunks_df (pd.DataFrame): DataFrame containing chunk details.

    Returns:
    str: Formatted string of the top 5 chunks with their page numbers and texts.
    """
    #torch.set_printoptions(precision=10)
    query_embedding = torch.tensor(vectoriser(query)).view(1, -1)
    loaded_embeddings
    cosine_similarity_scores = F.cosine_similarity(query_embedding, loaded_embeddings)
    print(cosine_similarity_scores)
    top_k = torch.topk(
        cosine_similarity_scores,
        k=20 if len(cosine_similarity_scores) >= 20 else len(cosine_similarity_scores),
    )
    top_chunks = top_k.indices.tolist()
    retrieved_chunks_df = chunks_df.iloc[top_chunks]
    chunk_final = "\n\n".join(
        f"Chunk Page Number: {row['Page_Number']}\n\n"
        f"Chunk Text: {row['Chunk_Text']}\n\n"
        "---- END OF CHUNK ----"
        for index, row in retrieved_chunks_df.iterrows()
    )

    print(top_chunks)
    print(chunk_final)
    return chunk_final


if __name__ == "__main__":
    # pages, ocr_status = read_pdf(
    #     "/Users/rohitsaluja/Documents/Github-silo-ai/RightHub/T2F-stlit/temp_data/NPL- D2 Document.pdf"
    # )
    # file_uuid = save_embeds_metadata(pages, ocr_status)
    # df, loaded_embeddings = embeds_metadata_loader(file_uuid)

    # query = "Who was the author supervised by?"

    # retreived_chunks = retreiver(query, loaded_embeddings, df)
    # print(retreived_chunks)

    # pages = data_chunker("temp_data/managing_ai_risks.pdf")
    # hunk_texts,chunk_page,chunk_embeds = embeds_metadata(pages)
    # save_embeds_metadata()

    # indexer(pages)

    # print(azure_ocr("temp_data/US2023214776A1 (2).pdf"))

    # pages, status= read_pdf("/Users/rohitsaluja/Documents/Github-silo-ai/RightHub/T2F-stlit/temp_data/US2023214776A1 (2).pdf")
    # indexer(pages, status)

    #print(vectoriser("This is a test string"))

    #read_pdf("temp_data/Invoice-A4012B25-0003 (1).pdf")

    #print(len(vectoriser("This is hello world")))

    #print(azure_ocr_to_vectorize("temp_data/US20150317431A1 (1).pdf"))

    print(check_ocr_to_vectorize("temp_data/US20150317431A1 (1).pdf"))

    # print(data_chunker("temp_data/33.161409-2022.11.23-21753929-1203-ocr.pdf"))
    # print(azure_ocr_to_vectorize("temp_data/US20150317431A1 (1).pdf"))