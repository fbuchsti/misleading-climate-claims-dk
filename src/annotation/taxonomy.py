"""
Defines the expected structured output format from the CARDS API.
Using Pydantic ensures typed, validated responses.
"""

from pydantic import BaseModel
from typing import List


class Category(BaseModel):
    """
    Represents a single CARDS taxonomy category.
    """
    category_number: str
    category_name: str


class Categories(BaseModel):
    """
    Represents the full CARDS response.
    Contains a list of detected categories.
    """
    categories: List[Category]