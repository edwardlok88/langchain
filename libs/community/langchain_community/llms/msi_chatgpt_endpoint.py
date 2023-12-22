import json
import urllib.request
import warnings
from abc import abstractmethod
from typing import Any, Dict, List, Mapping, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import BaseModel, validator
from langchain_core.utils import get_from_dict_or_env


class MSIChatGPTEndpointClient(object):
    """MSI ChatGPT Endpoint client."""

    def __init__(
        self, endpoint_url: str, endpoint_api_key: str, deployment_name: str = ""
    ) -> None:
        """Initialize the class."""
        if not endpoint_api_key or not endpoint_url:
            raise ValueError(
                """A key/token and REST endpoint should 
                be provided to invoke the endpoint"""
            )
        self.endpoint_url = endpoint_url
        self.endpoint_api_key = endpoint_api_key
        self.deployment_name = deployment_name

    def call(self, body: bytes, **kwargs: Any) -> bytes:
        """call."""

        # The azureml-model-deployment header will force the request to go to a
        # specific deployment. Remove this header to have the request observe the
        # endpoint traffic rules.
        headers = {
            "Content-Type": "application/json",
            "Authorization": ("Bearer " + self.endpoint_api_key),
        }
        req = urllib.request.Request(self.endpoint_url, body, headers)
        response = urllib.request.urlopen(req, timeout=kwargs.get("timeout", 50))
        result = response.read()
        return result


class ContentFormatterBase:
    """Transform request and response of endpoint to match with
    required schema.
    """

    """
    Example:
        .. code-block:: python
        
            class ContentFormatter(ContentFormatterBase):
                content_type = "application/json"
                accepts = "application/json"
                
                def format_request_payload(
                    self, 
                    prompt: str, 
                    model_kwargs: Dict
                ) -> bytes:
                    input_str = json.dumps(
                        {
                            "inputs": {"input_string": [prompt]}, 
                            "parameters": model_kwargs,
                        }
                    )
                    return str.encode(input_str)
                    
                def format_response_payload(self, output: str) -> str:
                    response_json = json.loads(output)
                    return response_json[0]["0"]
    """
    content_type: Optional[str] = "application/json"
    """The MIME type of the input data passed to the endpoint"""

    accepts: Optional[str] = "application/json"
    """The MIME type of the response data returned from the endpoint"""

    @staticmethod
    def escape_special_characters(prompt: str) -> str:
        """Escapes any special characters in `prompt`"""
        escape_map = {
            "\\": "\\\\",
            '"': '\\"',
            "\b": "\\b",
            "\f": "\\f",
            "\n": "\\n",
            "\r": "\\r",
            "\t": "\\t",
        }

        # Replace each occurrence of the specified characters with escaped versions
        for escape_sequence, escaped_sequence in escape_map.items():
            prompt = prompt.replace(escape_sequence, escaped_sequence)

        return prompt

    @abstractmethod
    def format_request_payload(self, prompt: str, model_kwargs: Dict) -> bytes:
        """Formats the request body according to the input schema of
        the model. Returns bytes or seekable file like object in the
        format specified in the content_type request header.
        """

    @abstractmethod
    def format_response_payload(self, output: bytes) -> str:
        """Formats the response body according to the output
        schema of the model. Returns the data type that is
        received from the response.
        """


class GPT2ContentFormatter(ContentFormatterBase):
    """Content handler for GPT2"""

    def format_request_payload(self, prompt: str, model_kwargs: Dict) -> bytes:
        prompt = ContentFormatterBase.escape_special_characters(prompt)
        request_payload = json.dumps(
            {"inputs": {"input_string": [f'"{prompt}"']}, "parameters": model_kwargs}
        )
        return str.encode(request_payload)

    def format_response_payload(self, output: bytes) -> str:
        return json.loads(output)[0]["0"]


class OSSContentFormatter(GPT2ContentFormatter):
    """Deprecated: Kept for backwards compatibility

    Content handler for LLMs from the OSS catalog."""

    content_formatter: Any = None

    def __init__(self) -> None:
        super().__init__()
        warnings.warn(
            """`OSSContentFormatter` will be deprecated in the future. 
                      Please use `GPT2ContentFormatter` instead.  
                      """
        )


class HFContentFormatter(ContentFormatterBase):
    """Content handler for LLMs from the HuggingFace catalog."""

    def format_request_payload(self, prompt: str, model_kwargs: Dict) -> bytes:
        ContentFormatterBase.escape_special_characters(prompt)
        request_payload = json.dumps(
            {"inputs": [f'"{prompt}"'], "parameters": model_kwargs}
        )
        return str.encode(request_payload)

    def format_response_payload(self, output: bytes) -> str:
        return json.loads(output)[0]["generated_text"]


class DollyContentFormatter(ContentFormatterBase):
    """Content handler for the Dolly-v2-12b model"""

    def format_request_payload(self, prompt: str, model_kwargs: Dict) -> bytes:
        prompt = ContentFormatterBase.escape_special_characters(prompt)
        request_payload = json.dumps(
            {
                "input_data": {"input_string": [f'"{prompt}"']},
                "parameters": model_kwargs,
            }
        )
        return str.encode(request_payload)

    def format_response_payload(self, output: bytes) -> str:
        return json.loads(output)[0]


class LlamaContentFormatter(ContentFormatterBase):
    """Content formatter for LLaMa"""

    def format_request_payload(self, prompt: str, model_kwargs: Dict) -> bytes:
        """Formats the request according to the chosen api"""
        prompt = ContentFormatterBase.escape_special_characters(prompt)
        request_payload = json.dumps(
            {
                "input_data": {
                    "input_string": [f'"{prompt}"'],
                    "parameters": model_kwargs,
                }
            }
        )
        return str.encode(request_payload)

    def format_response_payload(self, output: bytes) -> str:
        """Formats response"""
        return json.loads(output)[0]["0"]