from pydantic.main import BaseModel


class PromptCreatorModel(BaseModel):
    base_text: str = None
    base_replace_text: str = None
    prompt_key: str
    replacement_key: str = None
    replace_text: str = None


class MultiFileQAItems(BaseModel):
    plain_text_files_list: list[dict]
    question: str
