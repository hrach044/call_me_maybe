from enum import Enum
from pydantic import BaseModel, Field
from .schemas import FunctionSchema
import string


class State(Enum):
    EXPECT_NAME_PREFIX = "expect_name_prefix"
    EXPECT_FUNCTION_NAME = "expect_function_name"
    EXPECT_PARAMETERS_PREFIX = "expect_parameters_prefix"
    EXPECT_KEY = "expect_key"
    INSIDE_KEY = "inside_key"
    EXPECT_COLON = "expect_colon"
    EXPECT_VALUE = "expect_value"
    INSIDE_STRING_VALUE = "inside_string_value"
    INSIDE_NUMBER_VALUE = "inside_number_value"
    INSIDE_BOOLEAN_VALUE = "inside_boolean_value"
    EXPECT_COMMA_OR_CLOSE = "expect_comma_or_close"
    EXPECT_FINAL_CLOSE = "expect_final_close"
    DONE = "done"


class DecodingContext(BaseModel):
    current_state: State = State.EXPECT_NAME_PREFIX
    all_functions: dict[str, FunctionSchema] = Field(default_factory=dict)
    built_text: str = ""
    current_fragment: str = ""
    current_key: str | None = None
    remaining_params: set[str] = Field(default_factory=set)
    chosen_function: FunctionSchema | None = None


def get_legal_next_chars(context: DecodingContext) -> set[str]:
    if context.current_state == State.EXPECT_NAME_PREFIX:
        literal = "{\"name\":\""
        return {literal[len(context.current_fragment)]}
    elif context.current_state == State.EXPECT_FUNCTION_NAME:
        candidates = [p for p in context.all_functions
                      if p.startswith(context.current_fragment)]
        next_chars = set()
        for candidate in candidates:
            if len(candidate) != len(context.current_fragment):
                next_chars.add(candidate[len(context.current_fragment)])
            else:
                next_chars.add("\"")
        return next_chars
    elif context.current_state == State.EXPECT_PARAMETERS_PREFIX:
        literal = ",\"parameters\":{"
        return {literal[len(context.current_fragment)]}
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
        assert context.current_key is not None
        chars = {str(n) for n in range(10)} | {"."}
        if context.remaining_params - {context.current_key}:
            chars.add(",")
        else:
            chars.add("}")
        return chars
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
        if len(context.remaining_params) == 0:
            return {"}"}
        else:
            return {","}
    elif context.current_state == State.EXPECT_FINAL_CLOSE:
        return {"}"}
    else:
        raise ValueError(f"Unhandled state: {context.current_state}")


def apply_char(context: DecodingContext, char: str) -> DecodingContext:
    if context.current_state == State.EXPECT_NAME_PREFIX:
        literal = "{\"name\":\""
        new_fragment = context.current_fragment + char
        if new_fragment == literal:
            return DecodingContext.model_construct(
                current_state=State.EXPECT_FUNCTION_NAME,
                all_functions=context.all_functions,
                built_text=context.built_text + char,
                current_fragment="",
                current_key=context.current_key,
                 remaining_params=context.remaining_params,
                chosen_function=context.chosen_function,
            )
        else:
            return DecodingContext.model_construct(
                current_state=State.EXPECT_NAME_PREFIX,
                all_functions=context.all_functions,
                built_text=context.built_text + char,
                current_fragment=context.current_fragment + char,
                current_key=context.current_key,
                remaining_params=context.remaining_params,
                chosen_function=context.chosen_function,
            )
    elif context.current_state == State.EXPECT_FUNCTION_NAME:
        if char == "\"":
            chosen = context.all_functions[context.current_fragment]
            return DecodingContext.model_construct(
                current_state=State.EXPECT_PARAMETERS_PREFIX,
                all_functions=context.all_functions,
                built_text=context.built_text + char,
                chosen_function=chosen,
                current_fragment="",
                current_key=context.current_key,
                remaining_params=set(chosen.parameters),
            )
        else:
            return DecodingContext.model_construct(
                current_state=State.EXPECT_FUNCTION_NAME,
                all_functions=context.all_functions,
                built_text=context.built_text + char,
                current_fragment=context.current_fragment + char,
                current_key=context.current_key,
                remaining_params=context.remaining_params,
                chosen_function=context.chosen_function,
            )
    elif context.current_state == State.EXPECT_PARAMETERS_PREFIX:
        literal = ",\"parameters\":{"
        new_fragment = context.current_fragment + char
        if new_fragment == literal:
            return DecodingContext.model_construct(
                current_state=State.EXPECT_KEY,
                all_functions=context.all_functions,
                built_text=context.built_text + char,
                current_fragment="",
                current_key=None,
                remaining_params=context.remaining_params,
                chosen_function=context.chosen_function,
            )
        else:
            return DecodingContext.model_construct(
                current_state=State.EXPECT_PARAMETERS_PREFIX,
                all_functions=context.all_functions,
                built_text=context.built_text + char,
                current_fragment=context.current_fragment + char,
                current_key=context.current_key,
                remaining_params=context.remaining_params,
                chosen_function=context.chosen_function,
            )
    elif context.current_state == State.EXPECT_KEY:
        return DecodingContext.model_construct(
            current_state=State.INSIDE_KEY,
            all_functions=context.all_functions,
            built_text=context.built_text + char,
            current_fragment="",
            current_key=context.current_key,
            remaining_params=context.remaining_params,
            chosen_function=context.chosen_function,
        )
    elif context.current_state == State.INSIDE_KEY:
        if char == '"':
            return DecodingContext.model_construct(
                current_state=State.EXPECT_COLON,
                all_functions=context.all_functions,
                built_text=context.built_text + char,
                current_fragment="",
                current_key=context.current_fragment,
                remaining_params=context.remaining_params,
                chosen_function=context.chosen_function,
            )
        else:
            return DecodingContext.model_construct(
                current_state=State.INSIDE_KEY,
                all_functions=context.all_functions,
                built_text=context.built_text + char,
                current_fragment=context.current_fragment + char,
                current_key=context.current_key,
                remaining_params=context.remaining_params,
                chosen_function=context.chosen_function,
            )
    elif context.current_state == State.EXPECT_COLON:
        return DecodingContext.model_construct(
            current_state=State.EXPECT_VALUE,
            all_functions=context.all_functions,
            built_text=context.built_text + char,
            current_fragment="",
            current_key=context.current_key,
            remaining_params=context.remaining_params,
            chosen_function=context.chosen_function,
        )
    elif context.current_state == State.EXPECT_VALUE:
        assert context.chosen_function is not None
        assert context.current_key is not None
        param_type = (
                    context.chosen_function.parameters[context.current_key].type
                )
        if param_type == "number":
            return DecodingContext.model_construct(
                current_state=State.INSIDE_NUMBER_VALUE,
                all_functions=context.all_functions,
                built_text=context.built_text + char,
                current_fragment=context.current_fragment + char,
                current_key=context.current_key,
                remaining_params=context.remaining_params,
                chosen_function=context.chosen_function, 
        )
        elif param_type == "string":
            return DecodingContext.model_construct(
                current_state=State.INSIDE_STRING_VALUE,
                all_functions=context.all_functions,
                built_text=context.built_text + char,
                current_fragment="",
                current_key=context.current_key,
                remaining_params=context.remaining_params,
                chosen_function=context.chosen_function,
            )
        elif param_type == "boolean":
            return DecodingContext.model_construct(
                current_state=State.INSIDE_BOOLEAN_VALUE,
                all_functions=context.all_functions,
                built_text=context.built_text + char,
                current_fragment=context.current_fragment + char,
                current_key=context.current_key,
                remaining_params=context.remaining_params,
                chosen_function=context.chosen_function,
            )
        else:
            raise ValueError(f"Unhandled parameter type: {param_type}")
    elif context.current_state == State.INSIDE_NUMBER_VALUE:
        if char == ",":
            assert context.current_key is not None
            return DecodingContext.model_construct(
                current_state=State.EXPECT_KEY,
                all_functions=context.all_functions,
                built_text=context.built_text + char,
                current_fragment="",
                remaining_params=context.remaining_params - {context.current_key},
                current_key=None,
                chosen_function=context.chosen_function,
            )
        elif char == "}":
            assert context.current_key is not None
            return DecodingContext.model_construct(
                current_state=State.EXPECT_FINAL_CLOSE,
                all_functions=context.all_functions,
                built_text=context.built_text + char,
                current_fragment="",
                remaining_params=context.remaining_params - {context.current_key},
                current_key=None,
                chosen_function=context.chosen_function,
            )
        else:
            return DecodingContext.model_construct(
                current_state=State.INSIDE_NUMBER_VALUE,
                all_functions=context.all_functions,
                built_text=context.built_text + char,
                current_fragment=context.current_fragment + char,
                remaining_params=context.remaining_params,
                current_key=context.current_key,
                chosen_function=context.chosen_function,
            )
    elif context.current_state == State.INSIDE_STRING_VALUE:
        if char == '"':
            assert context.current_key is not None
            return DecodingContext.model_construct(
                current_state=State.EXPECT_COMMA_OR_CLOSE,
                all_functions=context.all_functions,
                built_text=context.built_text + char,
                current_fragment="",
                remaining_params=context.remaining_params - {context.current_key},
                current_key=None,
                chosen_function=context.chosen_function,
            )
        else:
            return DecodingContext.model_construct(
                current_state=State.INSIDE_STRING_VALUE,
                all_functions=context.all_functions,
                built_text=context.built_text + char,
                current_fragment=context.current_fragment + char,
                remaining_params=context.remaining_params,
                current_key=context.current_key,
                chosen_function=context.chosen_function,
            )
    elif context.current_state == State.INSIDE_BOOLEAN_VALUE:
        literal = ["true", "false"]
        new_fragment = context.current_fragment + char
        if new_fragment in literal:
            assert context.current_key is not None
            return DecodingContext.model_construct(
                current_state=State.EXPECT_COMMA_OR_CLOSE,
                all_functions=context.all_functions,
                built_text=context.built_text + char,
                current_fragment="",
                remaining_params=context.remaining_params - {context.current_key},
                current_key=None,
                chosen_function=context.chosen_function,
            )
        else:
            return DecodingContext.model_construct(
                current_state=State.INSIDE_BOOLEAN_VALUE,
                all_functions=context.all_functions,
                built_text=context.built_text + char,
                current_fragment=context.current_fragment + char,
                remaining_params=context.remaining_params,
                current_key=context.current_key,
                chosen_function=context.chosen_function,
            )
    elif context.current_state == State.EXPECT_COMMA_OR_CLOSE:
        if char == ",":
            return DecodingContext.model_construct(
                current_state=State.EXPECT_KEY,
                all_functions=context.all_functions,
                built_text=context.built_text + char,
                current_fragment=context.current_fragment,
                remaining_params=context.remaining_params,
                current_key=context.current_key,
                chosen_function=context.chosen_function,
            )
        else:
            return DecodingContext.model_construct(
                current_state=State.EXPECT_FINAL_CLOSE,
                all_functions=context.all_functions,
                built_text=context.built_text + char,
                current_fragment=context.current_fragment,
                remaining_params=context.remaining_params,
                current_key=context.current_key,
                chosen_function=context.chosen_function,
            )
    elif context.current_state == State.EXPECT_FINAL_CLOSE:
        return DecodingContext.model_construct(
            current_state=State.DONE,
            all_functions=context.all_functions,
            built_text=context.built_text + char,
            current_fragment=context.current_fragment,
            remaining_params=context.remaining_params,
            current_key=context.current_key,
            chosen_function=context.chosen_function,
        )
    else:
        raise ValueError(f"Unhandled state: {context.current_state}")

def is_token_legal(context: DecodingContext, token_str: str) -> DecodingContext | None:

    try:
        for char in token_str:
            if char not in get_legal_next_chars(context):
                return None
            context = apply_char(context, char)
    except ValueError:
        return None
    return context

def select_next_token(logits: list[float], vocab_strings: list[str],
    context: DecodingContext,) -> tuple[int, DecodingContext]:
    best_token_id = None
    best_score = -float("inf")
    best_context = None

    for token_id, score in enumerate(logits[:len(vocab_strings)]):
        token_str = vocab_strings[token_id]

        new_context = is_token_legal(context, token_str)

        if new_context is not None and score > best_score:
            best_score = score
            best_token_id = token_id
            best_context = new_context

    if best_token_id is None:
        raise RuntimeError("No legal tokens found for the current context!")
    
    return best_token_id, best_context
