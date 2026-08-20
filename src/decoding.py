from enum import Enum
from pydantic import BaseModel, Field
from .schemas import FunctionSchema
import string


class State(Enum):
    EXPECT_OPEN_BRACE = "expect_open_brace"
    EXPECT_KEY = "expect_key"
    INSIDE_KEY = "inside_key"
    EXPECT_COLON = "expect_colon"
    EXPECT_VALUE = "expect_value"
    INSIDE_STRING_VALUE = "inside_string_value"
    INSIDE_NUMBER_VALUE = "inside_number_value"
    INSIDE_BOOLEAN_VALUE = "inside_boolean_value"
    EXPECT_COMMA_OR_CLOSE = "expect_comma_or_close"


class DecodingContext(BaseModel):
    current_state: State = State.EXPECT_OPEN_BRACE
    built_text: str = ""
    current_fragment: str = ""
    current_key: str | None = None
    remaining_params: set[str] = Field(default_factory=set)
    chosen_function: FunctionSchema | None = None
    nesting_depth: int = 0


def get_legal_next_chars(context: DecodingContext) -> set[str]:
    if context.current_state == State.EXPECT_OPEN_BRACE:
        return {"{"}
    elif context.current_state == State.EXPECT_KEY:
        return {"\""}
    elif context.current_state == State.INSIDE_KEY:
        candidates = [p for p in context.remaining_params
                      if p.startswith(context.current_fragment)]
        next_chars = set()
        for candidate in candidates:
            if len(candidate) != len(context.current_fragment):
                next_chars.add(candidate[len(context.current_fragment)])
            else:
                next_chars.add("\"")
        return next_chars
    elif context.current_state == State.EXPECT_COLON:
        return {":"}
    elif context.current_state == State.EXPECT_VALUE:
        assert context.chosen_function is not None
        assert context.current_key is not None
        param_type = (
            context.chosen_function.parameters[context.current_key].type
        )
        if param_type == "number":
            return {str(n) for n in range(10)} | {"-"}
        elif param_type == "string":
            return {"\""}
        elif param_type == "boolean":
            return {"t", "f"}
        else:
            raise ValueError(f"Unhandled parameter type: {param_type}")
    elif context.current_state == State.INSIDE_STRING_VALUE:
        return set(string.printable)
    elif context.current_state == State.INSIDE_NUMBER_VALUE:
        return {str(n) for n in range(10)} | {".", ",", "}"}
    elif context.current_state == State.INSIDE_BOOLEAN_VALUE:
        candidates = [p for p in {"true", "false"}
                      if p.startswith(context.current_fragment)]
        next_chars = set()
        for candidate in candidates:
            if len(candidate) != len(context.current_fragment):
                next_chars.add(candidate[len(context.current_fragment)])
            else:
                next_chars.add(",")
                next_chars.add("}")
        return next_chars
    elif context.current_state == State.EXPECT_COMMA_OR_CLOSE:
        return {",", "}"}
    else:
        raise ValueError(f"Unhandled state: {context.current_state}")
