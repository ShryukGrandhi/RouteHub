import ast
import functools
import inspect
import re
from pathlib import Path

import routehub
from routehub._http import httpx

SKILL = Path(__file__).parents[1] / "skills" / "routehub" / "SKILL.md"
TEXT = SKILL.read_text()
BLOCKS = re.findall(r"```python\n(.*?)```", TEXT, re.S)
MOCK_REPLY = '{"title": "Single region", "severity": 3}'


def is_routehub(node):
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "routehub"
    )


def test_skill_names_and_arguments_exist():
    front = re.match(r"---\n(.*?)\n---\n", TEXT, re.S).group(1)
    assert re.search(r"^name: routehub$", front, re.M)
    assert re.search(r"^description: \S", front, re.M)
    assert len(TEXT.splitlines()) <= 300
    assert BLOCKS
    for block in BLOCKS:
        for node in ast.walk(ast.parse(block)):
            if is_routehub(node):
                assert node.attr in routehub.__all__, node.attr
            if isinstance(node, ast.Call) and is_routehub(node.func):
                function = getattr(routehub, node.func.attr)
                params = inspect.signature(function).parameters
                for keyword in node.keywords:
                    if keyword.arg is not None:
                        assert keyword.arg in params, (
                            f"{node.func.attr}({keyword.arg}=)"
                        )


def test_skill_examples_run_offline(monkeypatch, mock_openai):
    client, _ = mock_openai(
        httpx.Response(
            200,
            json={
                "object": "list",
                "model": "text-embedding-3-small",
                "data": [
                    {
                        "object": "embedding",
                        "index": 0,
                        "embedding": [0.1],
                    }
                ],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )
    )
    for name in ("completion", "acompletion"):
        mocked = functools.partial(
            getattr(routehub, name), mock_response=MOCK_REPLY
        )
        monkeypatch.setattr(routehub, name, mocked)
    monkeypatch.setattr(
        routehub,
        "embedding",
        functools.partial(routehub.embedding, client=client),
    )
    namespace = {}
    for number, block in enumerate(BLOCKS, 1):
        code = compile(
            block, f"SKILL.md python block {number}", "exec"
        )
        exec(code, namespace)
