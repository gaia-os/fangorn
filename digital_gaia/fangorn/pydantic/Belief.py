from pydantic import BaseModel
from typing import List


class Belief(BaseModel):
    mode: List[float]
    upper_limit: List[float]
    lower_limit: List[float]
