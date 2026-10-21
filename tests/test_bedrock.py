import pytest

import routehub as rh
from routehub import bedrock
from routehub._http import httpx


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
        assert request.headers["x-api-key"] == "bk-1"
        assert "authorization" not in request.headers
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
        assert result.usage.prompt_tokens == 5
        assert result.usage.completion_tokens == 2
        assert result.usage.total_tokens == 7
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
        assert recorder.requests[0].headers["x-api-key"] == "bk-1"

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

    def test_converse_url_quotes_model_ids(self):
        url = bedrock.converse_url(
            None, "us-east-1", "us.anthropic.claude-sonnet-4-6"
        )
        assert url == (
            "https://bedrock-runtime.us-east-1.amazonaws.com"
            "/model/us.anthropic.claude-sonnet-4-6/converse"
        )
