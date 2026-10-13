# CLAUDE.md

Guidance for coding agents and contributors working on RouteHub.

## What RouteHub is

A fast, lightweight LLM gateway with a litellm-compatible API, built on the OpenAI SDK. OpenAI-compatible providers are called through the OpenAI SDK. Claude is called through a native Messages API adapter and the results come back as OpenAI SDK objects.

## Module map

Everything lives in `routehub/`.

- `__init__.py`: lazy public names. `_LAZY` maps each name to its submodule, and the PEP 562 `__getattr__` imports it on first access.
- `main.py`: `completion`, `acompletion`, `embedding`, `aembedding`, `responses`, `aresponses`. Splits arguments into request parameters and gateway options (`_split_args`, `_build_call`), runs the OpenAI-compatible path, and holds the Anthropic send and retry loop (`_anthropic_send`).
- `batches.py`: `batched_completions` and `BatchJob`, which send chat requests through OpenAI's Batch API or Anthropic's Message Batches API. Each request is built by `main.py`'s `_split_args`, `_build_call` and `_openai_request` (or `_anthropic_prepare`), so a batch line matches a direct call.
- `providers.py`: the `PROVIDERS` table, provider aliases, and model-string resolution (`split_model`, `get_llm_provider`).
- `clients.py`: the client cache. LRU of 64 clients, keyed by every setting that changes the client and, for async clients, the running event loop. Sets pool limits and keep-alive (`keepalive_expiry`, default 60 seconds).
- `anthropic.py`: OpenAI chat to Messages translation (`build_request`, `to_chat_completion`) and the SSE `StreamTranslator`.
- `streaming.py`: `ChatStream` and `AsyncChatStream`, which yield `ChatCompletionChunk`s and map errors raised mid-stream.
- `exceptions.py`: litellm-named exceptions, `map_exception` and `exception_for_status`.
- `registry.py`: model metadata from OpenRouter's catalog, fetched on first use and cached for 5 minutes.
- `tokenizer.py`: `encode`, `decode` and `token_counter`, with tiktoken loaded on first use.
- `get_all_models.py`: lists models from each provider's own models endpoint.
- `utils.py`: the helpers litellm exposes under `litellm.utils`.
- `_http.py`: imports `httpx2` for openai 3.x and `httpx` for earlier versions.
- `_json.py`: orjson when installed (the `fast` extra), stdlib `json` otherwise.

## Commands

- `uv sync`: install the package and dev tools.
- `uv run pytest`: the offline suite, no API keys needed. Runs against the locked openai 3.x and httpx2.
- `uv run --with "openai<3" pytest`: the same suite against openai 2.x and httpx.
- `uv run --extra fast pytest`: the same suite with orjson installed.
- `uv run ruff check .`: lint.
- `uv run ruff format <paths>`: format the files you changed (line length 70). Leave files your change does not touch alone.

`tests/live_smoke.py` calls real providers and needs their API keys. pytest does not collect it. Never run it in CI or by default.

## Invariants: do not break these

- `import routehub` loads no third-party modules. New public names go in `_LAZY`. `openai`, `pydantic` and the HTTP library are imported inside the functions that need them. `exceptions.py` (and `streaming.py`, which imports it) loads the OpenAI SDK at import, so other modules import them inside functions too.
- No network calls at import time. The registry and `get_all_models` fetch on first use.
- No module-level configuration. Every setting is a per-call argument; provider keys and base URLs may also come from environment variables, read on each call. A setting that changes client behaviour must be part of the client cache key in `clients.py`.
- Return the OpenAI SDK's own types: `ChatCompletion`, `ChatCompletionChunk`, `CreateEmbeddingResponse`, and the SDK's `Response` (or its event stream) from `responses`. Don't re-wrap them.
- Core dependencies stay `openai`, `pydantic` and `tiktoken`. Anything else is an optional extra.
- Support openai 2.x (httpx) and 3.x (httpx2). Get the HTTP module with `from routehub._http import httpx`; never `import httpx` directly. Anything handed to a caller-supplied `client=` must work with that client's library: don't build a timeout or other object from `routehub._http` and pass it to a client that may come from the other library.
- Map provider errors to RouteHub exceptions. SDK and transport errors go through `map_exception`; error responses and stream error events on the native Anthropic path go through `exception_for_status`. Typed exceptions subclass the matching OpenAI SDK exception, so `except openai.RateLimitError` keeps working.
- Tests stay offline. Use the `mock_http` and `mock_openai` fixtures in `tests/conftest.py`, or `mock_response=`. Autouse fixtures already replace real API keys with fake ones (`fake_env`), serve a fake OpenRouter catalog (`catalog`) and zero the retry delays (`no_backoff`).

## Style

- Google-style docstrings on every function and method, with `Args:` and `Returns:` where they apply, in the existing `name (type): description` form.
- Comments only where the reason isn't obvious from the code.
- Ruff with line length 70 and the `E4`, `E7`, `E9`, `F`, `I` lint rules, configured in `pyproject.toml`.

## PR conventions

- Title: `[TYPE][area][Description]`, with TYPE one of `FIX`, `PERF`, `FEAT`, `CONF`, `DOCS`, and area the module or function changed. Examples: `[PERF][clients][Keep idle connections open longer between agent steps]`, `[FIX][map_exception][Raise a typed exception for an error event inside an OpenAI-compatible stream]`.
- Body starts with `Closes #N`.
- Body sections: Problem, Change, Verification. Verification includes test output on openai 2.x and 3.x (`uv run --with "openai<3" pytest` and `uv run pytest`).
