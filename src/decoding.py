"""Character-level JSON state machine for constrained decoding.

Tracks exactly which characters are legal next, given a function-call
JSON schema, so a language model's output can be restricted to only
schema-valid tokens at each generation step.
"""

from enum import Enum
from pydantic import BaseModel, Field
from .schemas import FunctionSchema
import string


class State(Enum):
    """Positions within the JSON function-call grammar being decoded."""

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
    """Immutable snapshot of decoding progress.

    Attributes:
        current_state: Current position in the JSON grammar.
        all_functions: Available function schemas, keyed by name.
        built_text: Full JSON text generated so far.
        current_fragment: Partial token being matched within the
            current state (e.g. a partial literal, key, or name).
        current_key: The parameter key currently being valued, if any.
        remaining_params: Parameter names not yet emitted for the
            chosen function.
        chosen_function: The function schema selected for this call,
            once known.
    """

    current_state: State = State.EXPECT_NAME_PREFIX
    all_functions: dict[str, FunctionSchema] = Field(default_factory=dict)
    built_text: str = ""
    current_fragment: str = ""
    current_key: str | None = None
    remaining_params: set[str] = Field(default_factory=set)
    chosen_function: FunctionSchema | None = None


def get_legal_next_chars(context: DecodingContext) -> set[str]:
    """Compute which characters are legal as the next character.

    Args:
        context: Current decoding state.

    Returns:
        The set of characters that would keep the generated text
        valid under the JSON function-call grammar.

    Raises:
        ValueError: If the context is in an unhandled state, or (for
            ``EXPECT_VALUE``) the parameter has an unhandled type.
    """

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
    """Advance the state machine by one legal character.

    Args:
        context: Current decoding state.
        char: The character to apply. Assumed to already be legal
            under ``get_legal_next_chars`` — this function does not
            re-validate it.

    Returns:
        A new ``DecodingContext`` reflecting the state after
        consuming ``char``.

    Raises:
        ValueError: If the context is in an unhandled state, or (for
            ``EXPECT_VALUE``) the parameter has an unhandled type.
    """

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
         context.chosen_function.parameters[context.current_key].type)
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
                remaining_params=context.remaining_params - {
                    context.current_key},
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
                remaining_params=context.remaining_params - {
                    context.current_key},
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
                remaining_params=context.remaining_params - {
                    context.current_key},
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
        literal2 = ["true", "false"]
        new_fragment = context.current_fragment + char
        if new_fragment in literal2:
            assert context.current_key is not None
            return DecodingContext.model_construct(
                current_state=State.EXPECT_COMMA_OR_CLOSE,
                all_functions=context.all_functions,
                built_text=context.built_text + char,
                current_fragment="",
                remaining_params=context.remaining_params - {
                    context.current_key},
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


def is_token_legal(context: DecodingContext,
                   token_str: str) -> DecodingContext | None:
    """Check whether a full token string is legal, char by char.

    Args:
        context: Current decoding state.
        token_str: The candidate token's decoded text.

    Returns:
        The resulting ``DecodingContext`` if every character in
        ``token_str`` is legal in sequence, otherwise ``None``.
    """

    try:
        for char in token_str:
            if char not in get_legal_next_chars(context):
                return None
            context = apply_char(context, char)
    except ValueError:
        return None
    return context


def is_token_legal_fast(context: DecodingContext,
                        token_str: str) -> DecodingContext | None:
    """Check token legality, with a fast path for string-value runs.

    Args:
        context: Current decoding state.
        token_str: The candidate token's decoded text.

    Returns:
        The resulting ``DecodingContext`` if ``token_str`` is legal,
        otherwise ``None``. Inside ``INSIDE_STRING_VALUE`` with no
        closing quote in the token, this skips per-character state
        transitions (every printable character is legal there) and
        appends the whole token at once; all other cases fall back to
        ``is_token_legal``.
    """

    if (
        context.current_state == State.INSIDE_STRING_VALUE
        and '"' not in token_str
    ):
        # No closing quote in this token -> definitely stays inside the string,
        # every char is legal (INSIDE_STRING_VALUE allows all
        # of string.printable).
        if not all(c in string.printable for c in token_str):
            return None
        return DecodingContext.model_construct(
            current_state=State.INSIDE_STRING_VALUE,
            all_functions=context.all_functions,
            built_text=context.built_text + token_str,
            current_fragment=context.current_fragment + token_str,
            current_key=context.current_key,
            remaining_params=context.remaining_params,
            chosen_function=context.chosen_function,
        )
    # Fall back to the careful, correct char-by-char path for everything else
    return is_token_legal(context, token_str)


def select_next_token(
    logits: list[float],
    vocab_strings: list[str],
    context: DecodingContext,
    first_char_to_token_ids: dict[str, list[int]],
) -> tuple[int, DecodingContext]:
    """Pick the highest-scoring legal next token.

    Args:
        logits: Raw next-token logits, indexed by token id.
        vocab_strings: Decoded string for each token id.
        context: Current decoding state.
        first_char_to_token_ids: Precomputed index mapping each
            character to the token ids whose decoded string starts
            with that character, used to avoid scanning the full
            vocabulary on every step.

    Returns:
        A tuple of the chosen token id and the resulting
        ``DecodingContext`` after applying it.

    Raises:
        RuntimeError: If no candidate token is legal in the current
            state.
    """

    best_token_id = None
    best_score = -float("inf")
    best_context = None

    legal_first_chars = get_legal_next_chars(context)
    candidate_ids: set[int] = set()
    for c in legal_first_chars:
        candidate_ids.update(first_char_to_token_ids.get(c, ()))

    for token_id in candidate_ids:
        score = logits[token_id]
        if score <= best_score:
            continue
        token_str = vocab_strings[token_id]
        new_context = is_token_legal_fast(context, token_str)
        if new_context is not None:
            best_score = score
            best_token_id = token_id
            best_context = new_context

    if best_token_id is None:
        raise RuntimeError("No legal tokens found for the current context!")

    assert best_context is not None
    return best_token_id, best_context
