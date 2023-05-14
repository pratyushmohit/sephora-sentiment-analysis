from typing import List

from pydantic import BaseModel


class PreprocessItem(BaseModel):
    filenames: List[str]