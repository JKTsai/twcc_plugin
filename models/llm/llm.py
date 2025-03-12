import json
import logging
from collections.abc import Generator
from typing import Optional, Union

import requests
import tiktoken
from dify_plugin import LargeLanguageModel
from dify_plugin.entities import I18nObject
from dify_plugin.entities.model import AIModelEntity, FetchFrom, ModelType
from dify_plugin.entities.model.llm import (
    LLMResult,
    LLMResultChunk,
    LLMResultChunkDelta,
)
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    PromptMessage,
    PromptMessageTool,
    UserPromptMessage,
)
from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
    InvokeAuthorizationError,
    InvokeBadRequestError,
    InvokeConnectionError,
    InvokeError,
    InvokeRateLimitError,
    InvokeServerUnavailableError,
)

logger = logging.getLogger(__name__)


class TwccLargeLanguageModel(LargeLanguageModel):
    """
    Model class for TWCC large language model.
    """

    API_URL = "https://api-ams.twcc.ai/api/models/conversation"

    def _invoke(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> Union[LLMResult, Generator]:
        """
        Invoke the model based on the response type (synchronous or streaming).
        """
        if stream:
            return self._handle_stream_response(
                model, credentials, prompt_messages, model_parameters, tools, stop, user
            )
        return self._handle_sync_response(
            model, credentials, prompt_messages, model_parameters, tools, stop, user
        )

    def _handle_stream_response(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        user: Optional[str] = None,
    ) -> Generator:
        """
        Handle streaming response from TWCC API.
        """
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": credentials.get("twcc_api_key"),
        }
        messages = self._convert_messages(prompt_messages)
        payload = {
            "model": model,
            "messages": messages,
            "parameters": model_parameters,
            "stream": True,
        }

        response = requests.post(
            self.API_URL, json=payload, headers=headers, stream=True
        )

        full_text = ""
        prompt_tokens = 0
        completion_tokens = 0

        final_chunk = LLMResultChunk(
            model=model,
            prompt_messages=prompt_messages,
            delta=LLMResultChunkDelta(
                index=0,
                message=AssistantPromptMessage(content=""),
            ),
        )
        for chunk in response.iter_lines():
            if chunk:
                chunk_str = chunk.decode("utf-8").strip()

                # Skip empty or invalid lines
                if not chunk_str.startswith("data:"):
                    continue

                try:
                    chunk_data = json.loads(chunk_str[5:])  # Remove "data:" prefix
                except json.JSONDecodeError:
                    continue

                text = chunk_data.get("generated_text", "")

                if text:
                    full_text += text
                    yield LLMResultChunk(
                        model=model,
                        prompt_messages=prompt_messages,
                        delta=LLMResultChunkDelta(
                            index=0, message=AssistantPromptMessage(content=text)
                        ),
                    )

                if (
                    "finish_reason" in chunk_data
                    and chunk_data["finish_reason"] is not None
                ):
                    final_chunk = LLMResultChunk(
                        model=model,
                        prompt_messages=prompt_messages,
                        delta=LLMResultChunkDelta(
                            index=0,
                            message=AssistantPromptMessage(content=text),
                            finish_reason=chunk_data["finish_reason"],
                        ),
                    )

            # Extract token usage from the last response
            if "prompt_tokens" in chunk_data:
                prompt_tokens = chunk_data.get("prompt_tokens", 0)
                completion_tokens = chunk_data.get("generated_tokens", 0)

        # if API didn't return promptt-tokens, calculate it from the first prompt message
        if not prompt_tokens:
            prompt_tokens = self.get_num_tokens(
                model, credentials, prompt_messages, tools
            )

        # if API didn't return completion tokens, calculate it from the full text
        if not completion_tokens:
            completion_tokens = self._get_num_tokens_from_str(
                model, credentials, full_text, tools
            )

        # transform usage
        usage = self._calc_response_usage(
            model, credentials, prompt_tokens, completion_tokens
        )

        final_chunk.delta.usage = usage

        yield final_chunk

    def _handle_sync_response(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        user: Optional[str] = None,
    ) -> LLMResult:
        """
        Handle synchronous response from TWCC API.
        """
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": credentials.get("twcc_api_key"),
        }

        messages = self._convert_messages(prompt_messages)
        payload = {
            "model": model,
            "messages": messages,
            "parameters": model_parameters,
        }

        try:
            response = requests.post(self.API_URL, json=payload, headers=headers)
            response.raise_for_status()  # Raise an error for HTTP error codes (4xx, 5xx)
        except requests.exceptions.RequestException as e:
            self._handle_invoke_error(e, response)

        data = response.json()
        assistant_message = AssistantPromptMessage(
            content=data.get("generated_text", "")
        )
        return LLMResult(
            model=model,
            prompt_messages=prompt_messages,
            message=assistant_message,
        )

    def _handle_invoke_error(
        self, exception: Exception, response: requests.Response = None
    ) -> None:
        """
        Handle TWCC API errors and map them to Dify InvokeError types.

        :param exception: The caught exception.
        :param response: The HTTP response object, if available.
        :raises InvokeError: The mapped error.
        """
        if isinstance(exception, requests.exceptions.ConnectionError):
            raise InvokeConnectionError("Failed to connect to TWCC API.") from exception
        elif isinstance(exception, requests.exceptions.Timeout):
            raise InvokeServerUnavailableError(
                "TWCC API timeout occurred."
            ) from exception

        if response is not None:
            if response.status_code == 400:
                raise InvokeBadRequestError(
                    "Invalid request parameters."
                ) from exception
            elif response.status_code == 401 or response.status_code == 403:
                raise InvokeAuthorizationError(
                    "Authentication failed. Check your API key."
                ) from exception
            elif response.status_code == 429:
                raise InvokeRateLimitError(
                    "API rate limit exceeded. Please try again later."
                ) from exception
            elif response.status_code >= 500:
                raise InvokeServerUnavailableError(
                    "TWCC API server is currently unavailable."
                ) from exception

        raise InvokeError(f"Unexpected error: {str(exception)}") from exception

    def _convert_messages(self, prompt_messages: list[PromptMessage]) -> list[dict]:
        """
        Convert PromptMessage list to TWCC API-compatible format.
        """
        messages = []
        roles = ["user", "assistant"]
        for index, message in enumerate(prompt_messages):
            role = roles[index % 2]
            messages.append({"role": role, "content": message.content})
        return messages

    def get_num_tokens(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        tools: Optional[list[PromptMessageTool]] = None,
    ) -> int:
        """
        Estimate the number of tokens in the given prompt messages using tiktoken.
        """
        encoding = tiktoken.get_encoding("cl100k_base")
        num_tokens = sum(
            len(encoding.encode(message.content)) for message in prompt_messages
        )
        return num_tokens

    def _get_num_tokens_from_str(
        self,
        model: str,
        credentials: dict,
        message: str,
        tools: Optional[list[PromptMessageTool]] = None,
    ) -> int:
        """
        Estimate the number of tokens in the given prompt messages using tiktoken.
        """
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(message))

    def validate_credentials(self, model: str, credentials: dict) -> None:
        """
        Validate TWCC API credentials by making a test API request with a simple prompt.
        """
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": credentials.get("twcc_api_key"),
        }

        payload = {"model": model, "prompt": "ping", "max_tokens": 10}

        try:
            response = requests.post(self.API_URL, json=payload, headers=headers)

            if response.status_code != 200:
                raise CredentialsValidateFailedError(
                    f"Invalid credentials for TWCC API: {response.status_code}, {response.text}"
                )
        except Exception as ex:
            raise CredentialsValidateFailedError(str(ex)) from ex

    def get_customizable_model_schema(
        self, model: str, credentials: dict
    ) -> AIModelEntity:
        """
        Define and return the schema for the TWCC model.
        """
        return AIModelEntity(
            model=model,
            label=I18nObject(zh_Hans=model, en_US=model),
            model_type=ModelType.LLM,
            features=[],
            fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
            model_properties={},
            parameter_rules=[],
        )

    @property
    def _invoke_error_mapping(self) -> dict[type[InvokeError], list[type[Exception]]]:
        """
        Map TWCC API errors to unified InvokeError types.

        :return: A mapping of invoke errors.
        """
        return {
            InvokeConnectionError: [
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
            ],
            InvokeServerUnavailableError: [requests.exceptions.HTTPError],
            InvokeRateLimitError: [requests.exceptions.RequestException],
            InvokeAuthorizationError: [requests.exceptions.RequestException],
            InvokeBadRequestError: [requests.exceptions.RequestException],
        }
