import os
from dotenv import load_dotenv
import yaml
import tiktoken
import subprocess
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient, AnalysisFeature
from PyPDF2 import PdfReader
from llama_index.core import SimpleDirectoryReader, ServiceContext, VectorStoreIndex, load_index_from_storage, \
    StorageContext, Document
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceWindowNodeParser

load_dotenv()
dirname = os.path.dirname(os.path.abspath(__file__))
data_dir_path = os.path.join(dirname, "data")

azure_ocr_endpoint = os.environ.get("AZURE_OCR_ENDPOINT")
azure_ocr_key = os.environ.get("AZURE_SECRET_KEY")


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


def build_sentence_window_index(documents,
                                llm,
                                embed_model=OpenAIEmbedding(),
                                sentence_window_size=3,
                                save_dir="sentence_index"):
    # Create sentence window node parser w/ default settings
    # Splits docs into sentences & augments each sentence w/ surrounding context
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=sentence_window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    # Create Service Context (wrapper object contains context needed for indexing)
    sentence_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        node_parser=node_parser,
    )
    # Create VectorStoreIndex using the Service Context (splits docs into sentences
    # & augments each sentence w/ surrounding context), then embeds & stores in VectorStore
    # Checks if index exists; if yes, loads it; if not creates it
    if not os.path.exists(save_dir):
        sentence_index = VectorStoreIndex.from_documents(
            documents, service_context=sentence_context
        )
        sentence_index.storage_context.persist(persist_dir=save_dir)
    else:
        sentence_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=sentence_context,
        )

    return sentence_index


def get_sentence_window_query_engine(sentence_index,
                                     similarity_top_k=10,
                                     rerank_top_n=3):
    # Define postprocessors
    # BAAI/bge-reranker-large
    # link: https://huggingface.co/BAAI/bge-reranker-large
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-large"
    )

    sentence_window_engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank], streaming=True
    )
    return sentence_window_engine


def azure_ocr_llidx(input_file_path: str):
    """
    Performs OCR on a document using Azure's Document Analysis Client and returns text by page.

    Args:
        input_file_path (str): File path of the document for OCR.

    Returns:
        list of dict: Each dictionary contains 'page_content' (text of the page) and
                      'page_label', corresponding to each page in the document.

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
        elem_dict["page_label"] = i + 1
        file_text.append(elem_dict)

    return file_text


def convert_to_llidx_docs(input_file_path: str):
    """
    Checks whether a PDF is already OCRd

    If any page has less than 100 characters, it uses Azure's OCR service
    to extract the text & page numbers & then converts it to LlamaIndex Document format.
    Otherwise, it uses SimpleDirectoryReader to convert it to LlamaIndex Document format.

    :param input_file_path: The path to the PDF file.
    :type input_file_path: str
    :return: List of LlamaIndex Documents
    :rtype: list
    """
    reader = PdfReader(input_file_path)
    extract_pages = []

    for i in range(len(reader.pages)):
        extracted_page = reader.pages[i].extract_text()
        extract_pages.append(extracted_page)
    # sometimes even non-ocrd pages have some text in the header that is added later and ocrable.
    # this could give false positive while checking of OCR.
    # we set char count limit to 100 to guard against that false positive.
    less_than_100_chars = any(len(element) < 100 for element in extract_pages)

    if less_than_100_chars:
        file_text = azure_ocr_llidx(input_file_path)
        print(type(file_text))
        print(file_text)
        docs = []
        for page in file_text:
            docs.append(Document(text=page["page_content"], metadata={'page_label': page["page_label"]}))
        print(f'Docs from an Azure OCRd File: {docs}')
        print(docs[0].metadata)
        print(type(docs[0]))
        return docs
    else:
        docs = SimpleDirectoryReader(input_files=[input_file_path]).load_data()
        print(f'Docs from an OCRd File: {docs}')
        print(docs[0].metadata)
        print(type(docs[0]))
        return docs

