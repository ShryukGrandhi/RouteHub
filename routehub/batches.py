"""Chat completions through OpenAI's Batch API."""

import inspect
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, List

from routehub._json import dumps, loads
from routehub.clients import openai_client
from routehub.main import (
    DEFAULT_NUM_RETRIES,
    _build_call,
    _openai_client,
    _openai_request,
    _split_args,
    completion,
)
from routehub.providers import get_llm_provider

_BATCH_DONE = ("completed", "failed", "expired", "cancelled")


@contextmanager
def _mapped_errors(provider: str, model: str = "") -> Iterator[None]:
    """Re-raise SDK errors inside the block as RouteHub exceptions.

    Args:
        provider (str): Provider that was called.
        model (str): Model that was called.

    Yields:
        None: Control to the block.
    """
    from routehub.exceptions import map_exception

    try:
        yield
    except Exception as error:
        mapped = map_exception(error, provider, model)
        if mapped is error:
            raise
        raise mapped from error


@dataclass
class BatchJob:
    """A submitted batch, rebuildable from its provider and id.

    Args:
        provider (str): Provider the batch runs on, such as "openai".
        id (str): The provider's batch id.
        client (Any): An OpenAI client; built from environment
            variables when omitted.
    """

    provider: str
    id: str
    client: Any = None

    def __post_init__(self) -> None:
        """Build a client from environment variables when none is given."""
        if self.client is None:
            _, provider, key, base = get_llm_provider(
                self.provider, self.provider
            )
            self.client = openai_client(
                provider,
                key,
                base,
                is_async=False,
                max_retries=DEFAULT_NUM_RETRIES,
                ssl_verify=True,
            )

    def status(self) -> str:
        """Return the batch's current status.

        Returns:
            str: The provider's status, such as "in_progress" or
            "completed".
        """
        with _mapped_errors(self.provider):
            return self.client.batches.retrieve(self.id).status

    def cancel(self) -> str:
        """Ask the provider to stop the batch.

        Returns:
            str: The status after the request, usually "cancelling".
        """
        with _mapped_errors(self.provider):
            return self.client.batches.cancel(self.id).status

    def results(self) -> List[Any]:
        """Download the results of a finished batch.

        Returns:
            List[Any]: A ChatCompletion, or the RouteHub exception for a
            failed request, per request in submission order; None for a
            request with no result, such as one a cancelled batch never
            ran.
        """
        from openai.types.chat import ChatCompletion

        from routehub.exceptions import (
            BadRequestError,
            exception_for_status,
        )

        with _mapped_errors(self.provider):
            batch = self.client.batches.retrieve(self.id)
            if (
                batch.status == "failed"
                or batch.status not in _BATCH_DONE
            ):
                reasons = (batch.errors and batch.errors.data) or []
                raise BadRequestError(
                    f"Batch {self.id} has status {batch.status}. "
                    + " ".join(e.message or "" for e in reasons),
                    llm_provider=self.provider,
                )
            files = [batch.output_file_id, batch.error_file_id]
            texts = [
                self.client.files.content(f).text for f in files if f
            ]
        results: List[Any] = [None] * batch.request_counts.total
        for raw in (r for t in texts for r in t.splitlines()):
            line = loads(raw)
            response = line.get("response") or {}
            status = response.get("status_code", 500)
            body = response.get("body") or {}
            if status < 400:
                result = ChatCompletion.model_validate(body)
            else:
                error = line.get("error") or body.get("error") or {}
                result = exception_for_status(
                    status,
                    error.get("message", ""),
                    self.provider,
                    body=error,
                )
            results[int(line["custom_id"])] = result
        return results


def batched_completions(
    model: str,
    requests: List[dict],
    *,
    wait: bool = True,
    poll_interval: float = 30,
    timeout: float = 24 * 3600,
    **kwargs: Any,
) -> Any:
    """Run chat completions through OpenAI's Batch API.

    Args:
        model (str): OpenAI model name, such as "gpt-5.4-mini".
        requests (List[dict]): completion arguments for each request,
            such as messages and max_tokens.
        wait (bool): Poll until the batch finishes and return its
            results; when False, return a BatchJob at once.
        poll_interval (float): Seconds between status checks.
        timeout (float): Seconds to wait before raising Timeout; the
            batch keeps running and BatchJob can still collect it.
        **kwargs (Any): completion arguments shared by every request,
            such as api_key, temperature or client.

    Returns:
        Any: The results of BatchJob.results when wait is on,
        otherwise the BatchJob.
    """
    from routehub.exceptions import BadRequestError, Timeout

    if not requests:
        raise BadRequestError(
            "requests is required and cannot be empty", model=model
        )
    spec = inspect.signature(completion).parameters
    lines = []
    for i, request in enumerate(requests):
        args = {**kwargs, **request, "model": model}
        named = {
            n: args.pop(n, p.default)
            for n, p in spec.items()
            if n != "kwargs"
        }
        call = _build_call(*_split_args({**named, "kwargs": args}))
        body, extra_body = _openai_request(call)
        lines.append(
            {
                "custom_id": str(i),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {**body, **extra_body},
            }
        )
    if call.provider != "openai":
        raise BadRequestError(
            f"batched_completions supports openai, not {call.provider}.",
            llm_provider=call.provider,
            model=model,
        )
    with _mapped_errors(call.provider, model):
        client = _openai_client(call, is_async=False)
        upload = client.files.create(
            file=("batch.jsonl", b"\n".join(map(dumps, lines))),
            purpose="batch",
        )
        batch = client.batches.create(
            input_file_id=upload.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
    job = BatchJob(call.provider, batch.id, client)
    if not wait:
        return job
    deadline = time.monotonic() + timeout
    while job.status() not in _BATCH_DONE:
        if time.monotonic() > deadline:
            raise Timeout(
                f"Batch {job.id} is still running after {timeout}s.",
                call.provider,
                model,
            )
        time.sleep(poll_interval)
    return job.results()
