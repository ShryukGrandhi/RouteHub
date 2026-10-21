import time
import types

import pytest

import routehub as rh
from routehub import bedrock
from routehub._http import httpx
from routehub.exceptions import BadRequestError


@pytest.fixture(autouse=True)
def aws_region_env(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "us-east-1")


def converse_body(**over):
    body = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "hi there"}],
            }
        },
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": 5,
            "outputTokens": 2,
            "totalTokens": 7,
            "cacheReadInputTokens": 3,
            "cacheWriteInputTokens": 1,
        },
        "ResponseMetadata": {"RequestId": "r-1"},
        "model": "anthropic.claude-sonnet-4-6",
    }
    body.update(over)
    return body


def reply(status=200, body=None):
    return httpx.Response(status, json=body or converse_body())


class TestBedrockCompletion:
    def test_bearer_key_reaches_converse(
        self, mock_http, monkeypatch
    ):
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        client, recorder = mock_http(reply())
        result = rh.completion(
            model="bedrock/anthropic.claude-sonnet-4-6",
            messages=[{"role": "user", "content": "hello"}],
            api_key="bk-1",
            client=client,
        )
        request = recorder.requests[0]
        assert request.headers["authorization"] == "Bearer bk-1"
        assert "x-api-key" not in request.headers
        assert (
            request.url.host
            == "bedrock-runtime.us-east-1.amazonaws.com"
        )
        assert request.url.path.startswith(
            "/model/anthropic.claude-sonnet-4-6/converse"
        )
        assert result.choices[0].message.content == "hi there"
        assert result.choices[0].finish_reason == "stop"

    def test_usage_maps_with_cache_tokens(self, mock_http):
        client, recorder = mock_http(reply())
        result = rh.completion(
            model="bedrock/amazon.nova-lite-v1",
            messages=[{"role": "user", "content": "hello"}],
            api_key="bk-1",
            client=client,
        )
        assert result.usage.prompt_tokens == 9
        assert result.usage.completion_tokens == 2
        assert result.usage.total_tokens == 11
        assert result.usage.cache_read_input_tokens == 3
        assert result.usage.cache_creation_input_tokens == 1

    def test_region_comes_from_env_and_extra(
        self, mock_http, monkeypatch
    ):
        monkeypatch.setenv("AWS_REGION", "eu-central-1")
        client, recorder = mock_http(reply())
        rh.completion(
            model="bedrock/amazon.nova-lite-v1",
            messages=[{"role": "user", "content": "hi"}],
            api_key="bk-1",
            client=client,
        )
        assert (
            recorder.requests[0].url.host
            == "bedrock-runtime.eu-central-1.amazonaws.com"
        )
        client2, recorder2 = mock_http(reply())
        rh.completion(
            model="bedrock/amazon.nova-lite-v1",
            messages=[{"role": "user", "content": "hi"}],
            api_key="bk-1",
            client=client2,
            aws_region="ap-south-1",
        )
        assert (
            recorder2.requests[0].url.host
            == "bedrock-runtime.ap-south-1.amazonaws.com"
        )

    def test_request_maps_openai_to_converse(self, mock_http):
        client, recorder = mock_http(reply())
        rh.completion(
            model="bedrock/anthropic.claude-sonnet-4-6",
            messages=[
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "hello"},
            ],
            api_key="bk-1",
            client=client,
            max_tokens=128,
            temperature=0.2,
            stop=["\n\n"],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Weather lookup",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"}
                            },
                        },
                    },
                }
            ],
        )
        body = recorder.last_json
        assert body["system"] == [{"text": "Be brief."}]
        assert body["messages"] == [
            {
                "role": "user",
                "content": [{"text": "hello"}],
            }
        ]
        assert body["inferenceConfig"] == {
            "maxTokens": 128,
            "temperature": 0.2,
            "stopSequences": ["\n\n"],
        }
        spec = body["toolConfig"]["tools"][0]["toolSpec"]
        assert spec["name"] == "get_weather"
        assert spec["inputSchema"]["json"]["properties"] == {
            "city": {"type": "string"}
        }

    def test_tool_use_round_trips(self, mock_http):
        body = converse_body(
            output={
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "t-1",
                                "name": "get_weather",
                                "input": {"city": "Paris"},
                            }
                        }
                    ],
                }
            },
            stopReason="tool_use",
        )
        client, recorder = mock_http(reply(body=body))
        result = rh.completion(
            model="bedrock/anthropic.claude-sonnet-4-6",
            messages=[
                {"role": "user", "content": "weather in Paris?"}
            ],
            api_key="bk-1",
            client=client,
        )
        assert result.choices[0].finish_reason == "tool_calls"
        call = result.choices[0].message.tool_calls[0]
        assert call.id == "t-1"
        assert call.function.name == "get_weather"
        assert call.function.arguments == '{"city": "Paris"}'

        client2, recorder2 = mock_http(reply())
        rh.completion(
            model="bedrock/anthropic.claude-sonnet-4-6",
            messages=[
                {"role": "user", "content": "weather in Paris?"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "t-1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Paris"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "t-1",
                    "content": "sunny",
                },
            ],
            api_key="bk-1",
            client=client2,
        )
        followup = recorder2.last_json["messages"]
        assert followup[1]["content"][0]["toolUse"] == {
            "toolUseId": "t-1",
            "name": "get_weather",
            "input": {"city": "Paris"},
        }
        assert followup[2]["content"][0]["toolResult"] == {
            "toolUseId": "t-1",
            "content": [{"text": "sunny"}],
        }

    async def test_acompletion_awaits_the_same_request(
        self, mock_http
    ):
        client, recorder = mock_http(reply(), is_async=True)
        result = await rh.acompletion(
            model="bedrock/amazon.nova-lite-v1",
            messages=[{"role": "user", "content": "hello"}],
            api_key="bk-1",
            client=client,
        )
        assert result.choices[0].message.content == "hi there"
        assert (
            recorder.requests[0].headers["authorization"]
            == "Bearer bk-1"
        )

    def test_no_credentials_raise_authentication_error(
        self, mock_http, monkeypatch
    ):
        for name in (
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "BEDROCK_API_KEY",
        ):
            monkeypatch.delenv(name, raising=False)
        client, recorder = mock_http(reply())
        with pytest.raises(Exception) as err:
            rh.completion(
                model="bedrock/amazon.nova-lite-v1",
                messages=[{"role": "user", "content": "hi"}],
                client=client,
            )
        assert "AWS" in str(err.value)
        assert recorder.requests == []

    def test_error_status_maps_to_exception(self, mock_http):
        client, _ = mock_http(
            reply(status=400, body={"message": "bad model"})
        )
        with pytest.raises(Exception) as err:
            rh.completion(
                model="bedrock/amazon.nova-lite-v1",
                messages=[{"role": "user", "content": "hi"}],
                api_key="bk-1",
                client=client,
                num_retries=0,
            )
        assert "bad model" in str(err.value)

    def test_streaming_rejected_for_now(self, mock_http):
        client, _ = mock_http(reply())
        with pytest.raises(Exception) as err:
            rh.completion(
                model="bedrock/amazon.nova-lite-v1",
                messages=[{"role": "user", "content": "hi"}],
                api_key="bk-1",
                client=client,
                stream=True,
            )
        assert "streaming" in str(err.value).lower()

    def test_embeddings_rejected(self, mock_http):
        client, _ = mock_http(reply())
        with pytest.raises(Exception):
            rh.embedding(
                model="bedrock/amazon.nova-lite-v1",
                input=["hi"],
                api_key="bk-1",
                client=client,
            )

    def test_responses_api_rejected_before_any_request(
        self, mock_openai
    ):
        client, recorder = mock_openai(reply())
        with pytest.raises(BadRequestError):
            rh.responses(
                model="bedrock/amazon.nova-lite-v1",
                input="hi",
                api_key="bk-1",
                client=client,
            )
        assert recorder.requests == []

    def test_parallel_tool_results_share_one_user_turn(
        self, mock_http
    ):
        client, recorder = mock_http(reply())
        rh.completion(
            model="bedrock/anthropic.claude-sonnet-4-6",
            messages=[
                {
                    "role": "user",
                    "content": "weather in Paris and Rome?",
                },
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "t-1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Paris"}',
                            },
                        },
                        {
                            "id": "t-2",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Rome"}',
                            },
                        },
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "t-1",
                    "content": "sunny",
                },
                {
                    "role": "tool",
                    "tool_call_id": "t-2",
                    "content": "rainy",
                },
                {"role": "user", "content": "thanks"},
            ],
            api_key="bk-1",
            client=client,
        )
        messages = recorder.last_json["messages"]
        assert [m["role"] for m in messages] == [
            "user",
            "assistant",
            "user",
        ]
        results = messages[2]["content"]
        assert len(results) == 3
        assert results[1]["toolResult"]["toolUseId"] == "t-2"
        assert results[2] == {"text": "thanks"}

    def test_developer_and_list_system_content(self, mock_http):
        client, recorder = mock_http(reply())
        rh.completion(
            model="bedrock/anthropic.claude-sonnet-4-6",
            messages=[
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "Be brief."}
                    ],
                },
                {"role": "developer", "content": "Use metric."},
                {"role": "user", "content": "hello"},
            ],
            api_key="bk-1",
            client=client,
        )
        body = recorder.last_json
        assert body["system"] == [
            {"text": "Be brief."},
            {"text": "Use metric."},
        ]
        assert body["messages"] == [
            {"role": "user", "content": [{"text": "hello"}]}
        ]

    def test_stop_string_and_max_completion_tokens(self, mock_http):
        client, recorder = mock_http(reply())
        rh.completion(
            model="bedrock/anthropic.claude-sonnet-4-6",
            messages=[{"role": "user", "content": "hello"}],
            api_key="bk-1",
            client=client,
            stop="END",
            max_completion_tokens=50,
        )
        assert recorder.last_json["inferenceConfig"] == {
            "maxTokens": 50,
            "stopSequences": ["END"],
        }

    def test_tool_choice_without_tools_is_ignored(self, mock_http):
        client, recorder = mock_http(reply())
        rh.completion(
            model="bedrock/anthropic.claude-sonnet-4-6",
            messages=[{"role": "user", "content": "hello"}],
            api_key="bk-1",
            client=client,
            tool_choice="auto",
        )
        assert "toolConfig" not in recorder.last_json

    def test_unsupported_param_raises_unless_dropped(self, mock_http):
        client, recorder = mock_http(reply())
        with pytest.raises(BadRequestError) as err:
            rh.completion(
                model="bedrock/anthropic.claude-sonnet-4-6",
                messages=[{"role": "user", "content": "hello"}],
                api_key="bk-1",
                client=client,
                response_format={"type": "json_object"},
            )
        assert "response_format" in str(err.value)
        assert recorder.requests == []
        result = rh.completion(
            model="bedrock/anthropic.claude-sonnet-4-6",
            messages=[{"role": "user", "content": "hello"}],
            api_key="bk-1",
            client=client,
            response_format={"type": "json_object"},
            drop_params=True,
        )
        assert result.choices[0].message.content == "hi there"

    def test_missing_region_is_a_clear_error(
        self, mock_http, monkeypatch
    ):
        monkeypatch.delenv("AWS_REGION")
        monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
        client, recorder = mock_http(reply())
        with pytest.raises(BadRequestError) as err:
            rh.completion(
                model="bedrock/amazon.nova-lite-v1",
                messages=[{"role": "user", "content": "hi"}],
                api_key="bk-1",
                client=client,
            )
        assert "AWS_REGION" in str(err.value)
        assert recorder.requests == []

    def test_retry_is_signed_with_a_fresh_clock(
        self, mock_http, monkeypatch
    ):
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIDEXAMPLE")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
        ticks = iter([0, 3600])
        monkeypatch.setattr(
            bedrock,
            "time",
            types.SimpleNamespace(
                strftime=time.strftime,
                gmtime=lambda: time.gmtime(next(ticks)),
                time=time.time,
                sleep=time.sleep,
            ),
        )
        client, recorder = mock_http(
            reply(status=503, body={"message": "busy"}), reply()
        )
        rh.completion(
            model="bedrock/amazon.nova-lite-v1",
            messages=[{"role": "user", "content": "hi"}],
            client=client,
            num_retries=1,
        )
        first, second = recorder.requests
        assert (
            first.headers["x-amz-date"]
            != second.headers["x-amz-date"]
        )


class TestSigV4:
    def test_signature_is_deterministic(self):
        body = b'{"modelId": "m"}'
        headers = bedrock.sigv4_headers(
            "POST",
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/m/converse",
            body,
            "us-east-1",
            "AKIDEXAMPLE",
            "secret",
            session_token="tok",
            amz_date="20260101T000000Z",
        )
        again = bedrock.sigv4_headers(
            "POST",
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/m/converse",
            body,
            "us-east-1",
            "AKIDEXAMPLE",
            "secret",
            session_token="tok",
            amz_date="20260101T000000Z",
        )
        assert headers == again
        auth = headers["authorization"]
        assert auth.startswith(
            "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20260101/"
            "us-east-1/bedrock/aws4_request"
        )
        assert "Signature=" in auth
        assert headers["x-amz-security-token"] == "tok"
        assert (
            headers["x-amz-content-sha256"]
            == __import__("hashlib").sha256(body).hexdigest()
        )

    def test_sigv4_used_when_no_api_key(self, mock_http, monkeypatch):
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIDEXAMPLE")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
        client, recorder = mock_http(reply())
        result = rh.completion(
            model="bedrock/amazon.nova-lite-v1",
            messages=[{"role": "user", "content": "hi"}],
            api_key="bedrock",
            client=client,
        )
        assert result.choices[0].message.content == "hi there"
        auth = recorder.requests[0].headers["authorization"]
        assert auth.startswith("AWS4-HMAC-SHA256")

    def test_colon_model_id_matches_aws_signature(self):
        headers = bedrock.sigv4_headers(
            "POST",
            "https://bedrock-runtime.us-east-1.amazonaws.com"
            "/model/anthropic.claude-3-5-sonnet-20240620-v1%3A0/converse",
            b'{"modelId": "anthropic.claude-3-5-sonnet-20240620-v1:0"}',
            "us-east-1",
            "AKIDEXAMPLE",
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            amz_date="20260101T000000Z",
        )
        assert headers["authorization"] == (
            "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20260101/"
            "us-east-1/bedrock/aws4_request, "
            "SignedHeaders=host;x-amz-content-sha256;x-amz-date, "
            "Signature=0e39903b089c85eec3562677e9fdc9edb62ec203a23905352b5dc92af75af5dc"
        )

    def test_converse_url_quotes_model_ids(self):
        url = bedrock.converse_url(
            None, "us-east-1", "us.anthropic.claude-sonnet-4-6"
        )
        assert url == (
            "https://bedrock-runtime.us-east-1.amazonaws.com"
            "/model/us.anthropic.claude-sonnet-4-6/converse"
        )
