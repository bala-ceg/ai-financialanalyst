"""This module defines Pydantic models for this project.

These models are used mainly for the structured tool and LLM outputs.
Resources:
- https://docs.pydantic.dev/latest/concepts/models/
"""

from __future__ import annotations

from pydantic import BaseModel
from typing import Optional, Dict, List

class FinancialAnalystState(BaseModel):
    ticker: str
    price_data: Optional[List[Dict]] = None
    financial_data: Optional[Dict] = None
    technical_data: Optional[Dict] = None
    sentiment_data: Optional[Dict] = None
    analysis: Optional[str] = None
