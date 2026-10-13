import json

import openai
from conftest import chat_completion

import routehub as rh
from routehub._http import httpx

USER = [{"role": "user", "content": "Hi"}]


class TestBatches:
    def test_results_come_back_in_request_order(self, mock_openai):
        def batch(status, **fields):
            return httpx.Response(
                200,
                json={"id": "batch_1", "status": status, **fields},
            )

        def jsonl(*lines):
            return httpx.Response(
                200,
                content="".join(
                    json.dumps(line) + "\n" for line in lines
                ).encode(),
            )

        done = batch(
            "completed",
            output_file_id="file-out",
            error_file_id="file-err",
            request_counts={"total": 2, "completed": 1, "failed": 1},
        )
        client, recorder = mock_openai(
            httpx.Response(200, json={"id": "file-in"}),
            batch("validating"),
            batch("in_progress"),
            done,
            done,
            jsonl(
                {
                    "custom_id": "1",
                    "response": {
                        "status_code": 200,
                        "body": chat_completion(content="Second"),
                    },
                    "error": None,
                }
            ),
            jsonl(
                {
                    "custom_id": "0",
                    "response": {
                        "status_code": 400,
                        "body": {
                            "error": {
                                "message": "maximum context length is 8 tokens"
                            }
                        },
                    },
                    "error": None,
                }
            ),
        )
        results = rh.batched_completions(
            model="gpt-5.4-mini",
            requests=[
                {"messages": USER, "max_tokens": 5},
                {"messages": USER},
            ],
            poll_interval=0,
            client=client,
        )
        upload = recorder.requests[0].content
        assert b'"custom_id":"0"' in upload
        assert b'"max_completion_tokens":5' in upload
        assert recorder.requests[1].url.path == "/v1/batches"
        assert isinstance(results[0], rh.ContextWindowExceededError)
        assert isinstance(
            results[1], openai.types.chat.ChatCompletion
        )
        assert results[1].choices[0].message.content == "Second"
