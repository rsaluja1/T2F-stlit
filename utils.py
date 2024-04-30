import os
import yaml
import tiktoken
import subprocess
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient, AnalysisFeature
from PyPDF2 import PdfReader
from dotenv import load_dotenv

load_dotenv()
azure_ocr_endpoint = os.getenv("AZURE_OCR_ENDPOINT")
azure_ocr_key = os.getenv("AZURE_SECRET_KEY")


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

    extract_pages = []
    for i in range(len(reader.pages)):
        extracted_page = reader.pages[i].extract_text()
        extract_pages.append(extracted_page)

    less_than_100_chars = any(len(element) < 100 for element in extract_pages)

    if not less_than_100_chars:
        parsed_text = " ".join(extract_pages)
        return parsed_text
    else:
        extracted_text = azure_ocr(input_file_path)
        clean_extracted_text = extracted_text.replace("\0", "")
        return clean_extracted_text


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

    return results_dict["content"]


def docx_to_pdf(input_path: str) -> str | None:
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
