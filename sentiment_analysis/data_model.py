from typing import List

from pydantic import BaseModel


class PreprocessItem(BaseModel):
    filenames: List[str]

class TrainModelItem(BaseModel):
    embeddings: List[str]
    sequences: List[str]
    numerical_feature: str
    class_label: str

class InferenceItem(BaseModel):
    embeddings: List[str]
    sequences: List[str]
    numerical_feature: str