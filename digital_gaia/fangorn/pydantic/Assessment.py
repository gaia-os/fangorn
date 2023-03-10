from datetime import date
from pydantic import BaseModel
from typing import Dict
from .Belief import Belief


class Assessment(BaseModel):
    assessment_date: date
    efe: float
    beliefs: Dict[str, Belief]
    predictions: Dict[str, Belief]
