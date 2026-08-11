from pydantic import BaseModel
from typing import Dict


class ParameterSchema(BaseModel):
    """Describes the declared type of a single function parameter."""

    type: str


class FunctionSchema(BaseModel):
    """Describes a callable function: its name, description,
    parameters, and return type."""

    name: str
    description: str
    name: str
    description: str
    parameters: Dict[str, ParameterSchema]
    returns: ParameterSchema


class PromptSchema(BaseModel):
    """Represents a single natural-language prompt to be processed."""

    prompt: str


class FunctionCallResult(BaseModel):
    """Represents the final structured output:
    which function was called, with which argument values,
    for a given prompt."""

    prompt: str
    name: str
    parameters: Dict[str, int | float | str | bool]
