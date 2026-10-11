# Migrating from litellm

Function names and module paths match litellm, so most code only changes its imports. The tables below cover what does not carry over.

## Imports

| litellm | RouteHub |
|---|---|
| `pip install litellm` | `pip install routehub` |
| `import litellm`, `litellm.completion(...)` | `import routehub`, `routehub.completion(...)` |
| `from litellm import completion, acompletion, embedding` | `from routehub import completion, acompletion, embedding` |
| `from litellm.utils import get_model_info, supports_vision` | `from routehub.utils import get_model_info, supports_vision` |
| `from litellm.exceptions import AuthenticationError` | `from routehub.exceptions import AuthenticationError` |
| `from litellm import model_list, encode` | `from routehub import model_list, encode` |

## Settings

| litellm | RouteHub |
|---|---|
| `litellm.drop_params = True` | `completion(..., drop_params=True)` |
| `litellm.num_retries = 3` | `completion(..., num_retries=3)` |
| `litellm.ssl_verify = False` | `completion(..., ssl_verify=False)` |
| `litellm.set_verbose = True` | `completion(..., set_verbose=True)` |
| `litellm.request_timeout = 30` | `completion(..., timeout=30)` |
| `litellm.api_key = ...`, `litellm.api_base = ...` | `completion(..., api_key=..., api_base=...)`, or the provider's environment variable |

RouteHub reads no module attributes, so `routehub.drop_params = True` does nothing. To apply settings to every call, wrap the function once, for example `functools.partial(routehub.completion, num_retries=3, drop_params=True)`.

## Objects

| litellm | RouteHub |
|---|---|
| `ModelResponse` | `openai.types.chat.ChatCompletion` |
| `CustomStreamWrapper` | an iterator of `openai.types.chat.ChatCompletionChunk`, usable in `with` (async: `async for`, `async with`) |
| `EmbeddingResponse` | `openai.types.CreateEmbeddingResponse` |
| `response["choices"][0]["message"]["content"]` | `response.choices[0].message.content` |

Responses are pydantic models: use attribute access, or `response.model_dump()` for a dict. Dictionary access raises `TypeError`.

## Behaviour that differs

- litellm-only keyword arguments are accepted and ignored by `completion` and `acompletion`: `acompletion`, `additional_drop_params`, `allowed_openai_params`, `base_model`, `cache`, `caching`, `context_window_fallback_dict`, `deployment_id`, `fallbacks`, `input_cost_per_token`, `logger_fn`, `metadata`, `model_info`, `model_list`, `no-log`, `num_retries_per_request`, `output_cost_per_token`, `preset_cache_key`, `proxy_server_request`, `retry_policy`, and any name starting with `litellm_`. Fallbacks and response caching do not happen; handle them in your code.
- Any other unknown keyword argument is sent to the provider in the request body unless `drop_params=True`.
- Requests to OpenAI send `max_tokens` as `max_completion_tokens`.
- There is no `Router`, no callbacks, no proxy server and no `completion_cost`. For cost, multiply `response.usage` token counts by the prices from `get_model_info`.
- `get_model_info`, `model_cost` and `model_list` come from OpenRouter's live model list, not a bundled price file. Unknown models raise `ModelNotMappedError`; add them with `register_model`.
- Claude is called through Anthropic's native Messages API, so `cache_control`, extended thinking and cached-token usage work as on Anthropic.
