import json
from typing import Any, List
from langchain_core.exceptions import OutputParserException
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.output_parsers.json import parse_partial_json
from langchain_core.outputs import ChatGeneration, Generation

class JsonOutputAzureFunctionsParser(JsonOutputFunctionsParser):
    """Parse an output as the Json object."""

    def parse_result(self, result: List[Generation], *, partial: bool = False) -> Any:
        if len(result) != 1:
            raise OutputParserException(
                f"Expected exactly one result, but got {len(result)}"
            )
        generation = result[0]
        if not isinstance(generation, ChatGeneration):
            raise OutputParserException(
                "This output parser can only be used with a chat generation."
            )
        message = generation.message
        try:
            function_call = message.additional_kwargs["function_call"]
            # Convert None value to string to allow conversion to JSON.
            if function_call["arguments"] is None:
                function_call["arguments"] = ''
        except KeyError as exc:
            if partial:
                return None
            else:
                raise OutputParserException(f"Could not parse function call: {exc}")
        try:
            if partial:
                try:
                    if self.args_only:
                        return parse_partial_json(
                            function_call["arguments"], strict=self.strict
                        )
                    else:
                        return {
                            **function_call,
                            "arguments": parse_partial_json(
                                function_call["arguments"], strict=self.strict
                            ),
                        }
                except json.JSONDecodeError:
                    return None
            else:
                if self.args_only:
                    try:
                        return json.loads(
                            function_call["arguments"], strict=self.strict
                        )
                    except (json.JSONDecodeError, TypeError) as exc:
                        raise OutputParserException(
                            f"Could not parse function call data: {exc}"
                        )
                else:
                    try:
                        return {
                            **function_call,
                            "arguments": json.loads(
                                function_call["arguments"], strict=self.strict
                            ),
                        }
                    except (json.JSONDecodeError, TypeError) as exc:
                        raise OutputParserException(
                            f"Could not parse function call data: {exc}"
                        )
        except KeyError:
            return None
