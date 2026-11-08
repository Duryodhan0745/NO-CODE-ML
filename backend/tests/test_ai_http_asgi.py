"""HTTP-level AI authorization matrix without opening a restricted socket."""
import asyncio
import config
import httpx
import services
from ai.auth import issue_test_token
from main import app, ai_store, ai_orchestrator


def test_first_ai_message_without_summary_succeeds(monkeypatch):
    class Provider:
        def complete_json(self, _messages):
            return {
                "response": "Tell me about your dataset.",
                "intent_kind": "general",
                "requested_operation": None,
                "needs_clarification": False,
                "tool_name": None,
                "tool_arguments": {},
                "needs_approval": False,
            }

    async def send_message():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://asgi.test") as client:
            headers = {"Authorization": f"Bearer {issue_test_token('summary-regression-user')}"}
            conversation = await client.post("/api/ai/conversations", json={}, headers=headers)
            assert conversation.status_code == 200
            response = await client.post(
                f"/api/ai/conversations/{conversation.json()['id']}/messages",
                json={"content": "Hello"},
                headers=headers,
            )
            assert response.status_code == 200
            assert response.json()["assistant"]["content"] == "Tell me about your dataset."

    monkeypatch.setattr(ai_orchestrator, "provider", Provider())
    asyncio.run(send_message())


async def _run_matrix():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://asgi.test") as client:
        a = {"Authorization": f"Bearer {issue_test_token('asgi-user-a')}"}
        b = {"Authorization": f"Bearer {issue_test_token('asgi-user-b')}"}
        dataset = services.upload_dataset("asgi-auth.csv", b"x,target\n1,0\n2,1\n3,0\n4,1\n5,1\n6,0\n7,1\n8,0\n9,1\n10,0\n")
        workspace = services.create_workspace(dataset["id"])["id"]
        assert (await client.post(f"/api/ai/workspaces/{workspace}/claim", headers=a)).status_code == 200
        assert (await client.post(f"/api/ai/workspaces/{workspace}/claim", headers=b)).status_code == 403
        conversation = await client.post("/api/ai/conversations", json={"workspace_id": workspace}, headers=a)
        assert conversation.status_code == 200
        cid = conversation.json()["id"]
        assert (await client.get(f"/api/ai/conversations/{cid}/context", headers=a)).status_code == 200
        assert (await client.get(f"/api/ai/conversations/{cid}/context", headers=b)).status_code == 403
        assert (await client.post(f"/api/ai/conversations/{cid}/messages", json={"content": "private"}, headers=b)).status_code == 403
        approval = await client.post("/api/ai/approvals/training", json={"workspace_id": workspace, "configuration": {"target": "target"}, "approved_by": "asgi-user-a"}, headers=a)
        assert approval.status_code == 200
        assert (await client.post("/api/ai/approvals/training", json={"workspace_id": workspace, "configuration": {"target": "target"}, "approved_by": "asgi-user-b"}, headers=b)).status_code == 403
        tool = {"tool_name": "get_workspace", "arguments": {"workspace_id": workspace}, "idempotency_key": "asgi-matrix-key-1"}
        assert (await client.post("/api/ai/tools/execute", json=tool, headers=b)).status_code == 403
        assert (await client.post("/api/ai/tools/execute", json=tool, headers=a)).status_code == 200
        assert (await client.post("/api/ai/conversations", json={}, headers={"Authorization": "Bearer forged"})).status_code == 401
        assert (await client.post("/api/ai/conversations", json={})).status_code == 401
        return {"workspace_claim": True, "conversation_isolation": True, "approval_isolation": True, "tool_isolation": True, "token_rejection": True}


def test_two_user_ai_authorization_asgi_http():
    result = asyncio.run(_run_matrix())
    assert all(result.values())


def test_training_jobs_are_scoped_to_workspace_owner(monkeypatch):
    class JobManager:
        def start(self, workspace_id, owner_id):
            return {"job_id": "job-owner-check", "workspace_id": workspace_id, "owner_id": owner_id, "status": "INITIALIZING"}

        def get(self, job_id, owner_id):
            if job_id == "job-owner-check" and owner_id == "training-owner":
                return {"job_id": job_id, "status": "TRAINING"}
            return None

        def cancel(self, job_id, owner_id):
            if job_id == "job-owner-check" and owner_id == "training-owner":
                return {"job_id": job_id, "status": "CANCELLING"}
            return None

    async def verify_routes():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://asgi.test") as client:
            owner_headers = {"Authorization": f"Bearer {issue_test_token('training-owner')}"}
            stranger_headers = {"Authorization": f"Bearer {issue_test_token('training-stranger')}"}
            dataset = services.upload_dataset("training-owner.csv", b"x,target\n1,0\n2,1\n3,0\n4,1\n5,1\n6,0\n")
            workspace_id = services.create_workspace(dataset["id"])["id"]
            assert ai_store.claim_workspace(workspace_id, "training-owner")
            start = await client.post(f"/api/workspaces/{workspace_id}/training-jobs", headers=owner_headers)
            assert start.status_code == 200
            assert start.json()["job"]["status"] == "INITIALIZING"
            assert (await client.get("/api/training-jobs/job-owner-check", headers=stranger_headers)).status_code == 404
            assert (await client.post("/api/training-jobs/job-owner-check/cancel", headers=stranger_headers)).status_code == 404
            assert (await client.get("/api/training-jobs/job-owner-check", headers=owner_headers)).json()["job"]["status"] == "TRAINING"
            assert (await client.post("/api/training-jobs/job-owner-check/cancel", headers=owner_headers)).json()["job"]["status"] == "CANCELLING"

    monkeypatch.setenv("AI_AUTH_SECRET", "test-secret-with-more-than-thirty-two-characters")
    monkeypatch.setattr("main.training_jobs", JobManager())
    asyncio.run(verify_routes())
