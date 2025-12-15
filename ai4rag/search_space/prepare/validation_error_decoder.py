# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from typing import Literal

from pydantic_core import ErrorDetails

from ai4rag.search_space.src.exceptions import SearchSpaceValueError
from ai4rag.utils.constants import SearchSpaceValidationErrors

__all__ = ["validation_error_decoder"]


def _get_error_constrain_name_with_location(
    location: tuple[int | str, ...], skip_last_element: bool = True
) -> tuple[str, int]:
    """
    Retrieves exact object location where the ValidationError occured.

    Parameters
    ----------

    location : tuple[int | str, ...]
        Location of the ValidationError.

    skip_last_element : bool = True
        Set to True when we want to omit last location element.

    Returns
    -------
    tuple[str, int]
        Tuple with constrain attribute and position in the list if the parameter is a list.

    Example
    -------
    ```python
    location = ('chunking', 0, 'chunk_size')

    _get_error_constrain_name_with_location(location, skip_last_element=False)

    >>> (chunking.chunk_size, 0)
    ```
    """

    position = list((filter(lambda x: isinstance(x, int), location)))

    if not position:
        return str(location[0]), 0

    position_index = location.index(position[0])
    if skip_last_element:
        location_slice = slice(position_index + 1, -1)
    else:
        location_slice = slice(position_index + 1, None)

    message_constraints = location[:position_index] + location[location_slice]
    constrain_name = ".".join(message_constraints)
    return constrain_name, int(position[0])


def _get_parameter_length(
    validation_error: ErrorDetails, parameter_name: Literal["min_length", "max_length"]
) -> tuple[str, str]:
    """
    Retrieves actual parameter length from validation error and valid parameter length.

    Parameters
    ----------

    validation_error : ErrorDetails
        Dict like object which stores error information.

    parameter_name : Literal["min_length", "max_length"]
        Parameter name for which parameter length is retrieved.

    Returns
    -------
    tuple[str, str]
        Tuple which contains required min or max length of the parameter and the
        actual input length which caused the error.

    Example
    -------
    ```python
    error_details: ErrorDetails = {
        "type": "too_long",
        "loc": ("chunking",),
        "msg": "List should have at most 4 items after validation, not 5",
        "input": [
            {"chunk_size": 1000},
            {"chunk_size": 900},
            {"chunk_size": 800},
            {"chunk_size": 700},
            {"chunk_size": 600},
        ],
        "ctx": {"field_type": "List", "max_length": 4, "actual_length": 5},
        "url": "https://errors.pydantic.dev/2.9/v/too_long",
    }
    _get_parameter_length(error_details, "max_length")
    >>> (4, 5)
    ```
    """
    context = validation_error.get("ctx", {})
    required_items_num = f"{items} item" if (items := context.get(parameter_name)) == 1 else f"{items} items"
    input_items_num = context.get("actual_length", "")
    return required_items_num, input_items_num


def _get_parameter_limes_value_and_input(
    validation_error: ErrorDetails, parameter_name: Literal["le", "ge"]
) -> tuple[str, str]:
    """
    Retrieves parameter limes (upper or lower bound) and input which caused the validation error.

    Parameters
    ----------

    validation_error : ErrorDetails
        Dict like object which stores error information.

    parameter_name : Literal["le", "ge"]
        Parameter name for which parameter length is retrieved.

    Returns
    -------
    tuple[str, str]
        Tuple with upper or lower bound for the parameter value and the
        input value which caused the error.

    Example
    -------
    error_details: ErrorDetails = {
        'type': 'less_than_equal',
        'loc': ('chunking', 0, 'chunk_size'),
        'msg': 'Input should be less than or equal to 1024',
        'input': 50000, 'ctx': {'le': 1024},
        'url': 'https://errors.pydantic.dev/2.9/v/less_than_equal'
    }
    _get_parameter_limes_value_and_input(error_details, "le")
    >>> (1024, 50000)
    """
    context = validation_error.get("ctx", {})
    limes = str(context.get(parameter_name, 0))
    input_value = validation_error.get("input")
    return limes, input_value


def validation_error_decoder(
    validation_error: ErrorDetails,
) -> None:
    """
    Tiny wrapper for Pydantic ValidationError.
    Unpacks aforementioned error and raise it as SearchSpaceValueError.
    Every SearchSpaceValueError is re-raised from None to get rid of Pydantic traceback.

    Parameters
    ----------

    validation_error : ErrorDetails
        Dict like object which stores error information.

    Raises
    ------

    SearchSpaceValueError
        Context of this error depends on which ValidationError type occured during payload validation.

    Examples
    --------
    First example:

    ```python
    error_details: ErrorDetails = {
        'type': 'list_type', 'loc': ('chunking',),
        'msg': 'Input should be a valid list',
        'input': 'invalid_key',
        'url': 'https://errors.pydantic.dev/2.9/v/list_type'
    }
    validation_error_decoder(error_details) # match for SearchSpaceValidationErrors.LIST_TYPE
    ```

    Second example:
    ```python
    error_details: ErrorDetails = {
        'type': 'less_than_equal',
        'loc': ('chunking', 0, 'chunk_size'),
        'msg': 'Input should be less than or equal to 1024',
        'input': 50000, 'ctx': {'le': 1024},
        'url': 'https://errors.pydantic.dev/2.9/v/less_than_equal'
    }
    validation_error_decoder(error_details) # match for SearchSpaceValidationErrors.LESS_THAN_EQUAL
    ```
    """

    # if SearchSpaceValueError occurred during validation then it will be re-raised here
    if (context := validation_error.get("ctx")) and (error := context.get("error")):
        raise error from None

    location = validation_error.get("loc")

    match validation_error.get("type"):
        case SearchSpaceValidationErrors.UNEXPECTED_KEYWORD_ARGUMENT:

            if len(location) == 1:
                raise SearchSpaceValueError(f"{location[0]} is not a recognized parameter.") from None

            constrain_name, position = _get_error_constrain_name_with_location(location=location)
            raise SearchSpaceValueError(
                f"Constraint `{constrain_name}` at position {position} got invalid parameter name `{location[-1]}`."
            ) from None

        case SearchSpaceValidationErrors.LITERAL_ERROR:
            constrain_name, position = _get_error_constrain_name_with_location(
                location=location, skip_last_element=False
            )
            invalid_literal_value = validation_error.get("input")
            valid_literal_values = validation_error.get("ctx", {}).get("expected")
            raise SearchSpaceValueError(
                f"Constraint `{constrain_name}` at position {position} got invalid value `{invalid_literal_value}`, "
                f"but expected value is {valid_literal_values}."
            ) from None

        case SearchSpaceValidationErrors.TOO_SHORT:
            constrain_name = location[0]
            required_items_num, input_items_num = _get_parameter_length(
                validation_error=validation_error, parameter_name="min_length"
            )
            raise SearchSpaceValueError(
                f"Constraint `{constrain_name}` should have at least {required_items_num}, not `{input_items_num}`."
            ) from None

        case SearchSpaceValidationErrors.TOO_LONG:
            constrain_name = location[0]
            required_items_num, input_items_num = _get_parameter_length(
                validation_error=validation_error, parameter_name="max_length"
            )
            raise SearchSpaceValueError(
                f"Constraint `{constrain_name}` should have at most {required_items_num}, not `{input_items_num}`."
            ) from None

        case SearchSpaceValidationErrors.LESS_THAN_EQUAL:
            constrain_name, position = _get_error_constrain_name_with_location(
                location=location, skip_last_element=False
            )
            required_num_bound, input_num = _get_parameter_limes_value_and_input(
                validation_error=validation_error, parameter_name="le"
            )
            raise SearchSpaceValueError(
                f"Constraint `{constrain_name}` at position {position} should be less than or equal to "
                f"{required_num_bound}, but is equal to `{input_num}`."
            ) from None

        case SearchSpaceValidationErrors.GREATER_THAN_EQUAL:
            constrain_name, position = _get_error_constrain_name_with_location(
                location=location, skip_last_element=False
            )
            required_num_bound, input_num = _get_parameter_limes_value_and_input(
                validation_error=validation_error, parameter_name="ge"
            )
            raise SearchSpaceValueError(
                f"Constraint `{constrain_name}` at position {position} should be higher than or equal to "
                f"{required_num_bound}, but is equal to `{input_num}`."
            ) from None

        case (
            SearchSpaceValidationErrors.INT_FROM_FLOAT
            | SearchSpaceValidationErrors.INT_PARSING
            | SearchSpaceValidationErrors.INT_TYPE
            | SearchSpaceValidationErrors.LIST_TYPE
        ):
            constrain_name = ".".join([str(location_piece) for location_piece in location])
            constrain_input = validation_error.get("input")
            raise SearchSpaceValueError(
                f"Constraint {constrain_name} got invalid value {constrain_input}. "
                f"Check the docs for valid input constraints."
            ) from None

        case _:
            # it should not be raised in any case
            # if you will trigger the below error then wrap it up as shown above
            constrain_name = ".".join([str(location_piece) for location_piece in location])
            constrain_input = validation_error.get("input")
            raise SearchSpaceValueError(
                f"Unknown validation error occurred for constraint `{constrain_name}` for input {constrain_input}."
            ) from None
