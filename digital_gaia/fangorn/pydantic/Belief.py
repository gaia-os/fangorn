from pydantic import BaseModel
from typing import List


class Belief(BaseModel):
    mean: List[List[List[float]]]
    upper_limit: List[List[List[float]]]
    lower_limit: List[List[List[float]]]
