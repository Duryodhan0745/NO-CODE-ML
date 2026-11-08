import asyncio
import json

import httpx

import ai.provider as provider_module
from ai.auth import issue_test_token
from main import ai_orchestrator, ai_store, app


def _structured_response(text):
    return {
        "response": text,
        "intent_kind": "question",
        "requested_operation": None,
        "needs_clarification": False,
        "tool_name": None,
        "tool_arguments": {},
        "needs_approval": False,
    }


def test_provider_stream_decodes_response_in_chunks(monkeypatch):
    expected = _structured_response('A "quoted" answer with a code `block`.')
    content = json.dumps(expected)

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def __iter__(self):
            for start in range(0, len(content), 7):
                event = {"choices": [{"delta": {"content": content[start:start + 7]}}]}
                yield f"data: {json.dumps(event)}\n".encode()
            yield b"data: [DONE]\n"

    monkeypatch.setattr(provider_module.request, "urlopen", lambda *_args, **_kwargs: Response())
    provider = provider_module.OpenAICompatibleProvider(
        provider_module.ProviderConfig(api_key="test-key", model="test-model")
    )
    chunks = []
    result = provider.complete_json_stream([], chunks.append)

    assert result == expected
    assert "".join(chunks) == expected["response"]
    assert len(chunks) > 1


def test_streaming_http_route_persists_one_assistant_response(monkeypatch):
    class Provider:
        def complete_json_stream(self, _messages, on_text_chunk):
            on_text_chunk("Dataset ")
            on_text_chunk("summary.")
            return _structured_response("Dataset summary.")

    async def send():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://asgi.test") as client:
            token = issue_test_token("stream-user")
            headers = {"Authorization": f"Bearer {token}"}
            conversation = await client.post("/api/ai/conversations", json={}, headers=headers)
            assert conversation.status_code == 200
            conversation_id = conversation.json()["id"]
            response = await client.post(
                f"/api/ai/conversations/{conversation_id}/messages/stream",
                json={"content": "What is this dataset about?"},
                headers=headers,
            )
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            events = [json.loads(frame[6:]) for frame in response.text.split("\n\n") if frame.startswith("data: ")]
            assert [event["text"] for event in events if event["type"] == "delta"] == ["Dataset ", "summary."]
            assert events[-1]["type"] == "complete"
            assert events[-1]["assistant"]["content"] == "Dataset summary."
            context = await client.get(f"/api/ai/conversations/{conversation_id}/context", headers=headers)
            messages = context.json()["messages"]
            assert [message["role"] for message in messages] == ["user", "assistant"]

    monkeypatch.setenv("AI_AUTH_SECRET", "test-secret-with-more-than-thirty-two-characters")
    monkeypatch.setattr(ai_orchestrator, "provider", Provider())
    asyncio.run(send())
