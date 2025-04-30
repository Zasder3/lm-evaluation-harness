import json
import logging
import os
from functools import cached_property

import requests

from lm_eval.models.openai_completions import OpenAIChatCompletion


eval_logger = logging.getLogger(__name__)


class OpenRouterChatCompletion(OpenAIChatCompletion):
    """
    OpenRouter API client that handles whitespace-prefixed responses.

    This model extends the OpenAI Chat Completions API client and adds
    special handling for OpenRouter's responses which may contain leading whitespace.
    """

    def __init__(
        self,
        base_url="https://openrouter.ai/api/v1/chat/completions",
        tokenizer_backend=None,
        tokenized_requests=False,
        **kwargs,
    ):
        # Add HTTP-Referer if not provided in headers
        if "headers" not in kwargs:
            kwargs["headers"] = {"HTTP-Referer": "http://localhost:3000"}
        elif "HTTP-Referer" not in kwargs["headers"]:
            kwargs["headers"]["HTTP-Referer"] = "http://localhost:3000"

        super().__init__(
            base_url=base_url,
            tokenizer_backend=tokenizer_backend,
            tokenized_requests=tokenized_requests,
            **kwargs,
        )

    def model_call(
        self,
        messages,
        *,
        generate=True,
        gen_kwargs=None,
        **kwargs,
    ):
        """Override model_call to handle whitespace in responses"""
        try:
            response = requests.post(
                self.base_url,
                json=self._create_payload(
                    self.create_message(messages),
                    generate=generate,
                    gen_kwargs=gen_kwargs,
                    seed=self._seed,
                    eos=self.eos_string,
                    **kwargs,
                ),
                headers=self.header,
                verify=self.verify_certificate,
                timeout=self.timeout,
            )

            if not response.ok:
                eval_logger.warning(
                    f"API request failed with status {response.status_code}, error message: {response.text}. Retrying..."
                )
            response.raise_for_status()

            # Strip whitespace before parsing JSON
            cleaned_text = response.text.strip()
            eval_logger.debug(f"Raw response text: {response.text!r}")
            eval_logger.debug(f"Cleaned response text: {cleaned_text!r}")

            return json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            eval_logger.error(
                f"JSON decode error: {e} - Raw response: {response.text!r}"
            )
            raise
        except Exception as e:
            eval_logger.error(f"API request error: {e}")
            raise

    async def amodel_call(
        self,
        session,
        messages,
        *,
        generate=True,
        cache_keys=None,
        ctxlens=None,
        gen_kwargs=None,
        **kwargs,
    ):
        """Override amodel_call to handle whitespace in responses for async calls"""
        # Create payload
        payload = self._create_payload(
            self.create_message(messages),
            generate=generate,
            gen_kwargs=gen_kwargs,
            seed=self._seed,
            **kwargs,
        )
        cache_method = "generate_until" if generate else "loglikelihood"

        try:
            async with session.post(
                self.base_url,
                json=payload,
                headers=self.header,
            ) as response:
                if not response.ok:
                    error_text = await response.text()
                    eval_logger.warning(
                        f"API request failed with status {response.status_code}, error message: {error_text}. Retrying..."
                    )
                # Raising exception will retry the request
                response.raise_for_status()

                # Get response text and strip whitespace
                response_text = await response.text()
                cleaned_text = response_text.strip()
                eval_logger.debug(f"Raw response text: {response_text!r}")
                eval_logger.debug(f"Cleaned response text: {cleaned_text!r}")

                outputs = json.loads(cleaned_text)

            answers = (
                self.parse_generations(
                    outputs=outputs,
                )
                if generate
                else self.parse_logprobs(
                    outputs=outputs,
                    tokens=messages,
                    ctxlens=ctxlens,
                )
            )
            if cache_keys:
                for res, cache in zip(answers, cache_keys):
                    self.cache_hook.add_partial(cache_method, cache, res)
            return answers
        except json.JSONDecodeError as e:
            eval_logger.error(
                f"JSON decode error: {e} - Raw response: {response_text!r}"
            )
            raise
        except Exception as e:
            eval_logger.error(f"API request error: {e}")
            raise

    @cached_property
    def api_key(self):
        """Override this property to return the API key for the API request."""
        key = os.environ.get("OPENAI_API_KEY", None)
        if key is None:
            key = os.environ.get("OPENROUTER_API_KEY", None)
            if key is None:
                raise ValueError(
                    "API key not found. Please set either the `OPENAI_API_KEY` or `OPENROUTER_API_KEY` environment variable."
                )
        return key


# Register the model
from lm_eval.api.registry import register_model


register_model("openrouter-chat", "openrouter")(OpenRouterChatCompletion)
