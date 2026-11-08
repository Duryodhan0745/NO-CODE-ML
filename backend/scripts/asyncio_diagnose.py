"""Bounded diagnostics for the Windows async runtime used by this backend.

Run from backend: ``python scripts/asyncio_diagnose.py``.
Each probe runs in a child process so a platform-level hang cannot block later
probes. This file intentionally does not change application runtime behavior.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path


PROBES = {
    "tcp_loopback": """
        import socket, threading
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(('127.0.0.1', 0)); listener.listen(1)
        port = listener.getsockname()[1]
        received = []
        def accept_once():
            conn, _ = listener.accept(); received.append(conn.recv(2)); conn.sendall(b'ok'); conn.close()
        thread = threading.Thread(target=accept_once); thread.start()
        client = socket.create_connection(('127.0.0.1', port), timeout=2)
        client.sendall(b'hi'); assert client.recv(2) == b'ok'
        client.close(); thread.join(2); listener.close(); assert received == [b'hi']
        print('tcp-loopback-ok')
    """,
    "socketpair": """
        import socket
        a, b = socket.socketpair()
        a.sendall(b'ok'); assert b.recv(2) == b'ok'
        a.close(); b.close(); print('socketpair-ok')
    """,
    "asyncio_run": """
        import asyncio
        async def main(): return 'coroutine-ok'
        print(asyncio.run(main()))
    """,
    "loop_create_close": """
        import asyncio
        loop = asyncio.new_event_loop(); print(type(loop).__name__)
        loop.close(); print('loop-close-ok')
    """,
    "fastapi_import": """
        from app.main import app
        print(app.title)
    """,
    "asgi_transport": """
        import asyncio, httpx
        from app.main import app
        async def main():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
                response = await client.get('/api/v1/health')
                assert response.status_code == 200
                print(response.json())
        asyncio.run(main())
    """,
    "uvicorn_startup": """
        import uvicorn
        from app.main import app
        config = uvicorn.Config(app, host='127.0.0.1', port=0, lifespan='on', log_level='error')
        server = uvicorn.Server(config)
        print('uvicorn-config-ok', server.config.host)
    """,
}


def run_probe(name: str, source: str) -> bool:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    try:
        completed = subprocess.run(
            [sys.executable, "-X", "faulthandler", "-c", textwrap.dedent(source)],
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )
    except subprocess.TimeoutExpired as error:
        print(f"{name}: TIMEOUT after 10 seconds")
        if error.stdout:
            print(error.stdout)
        if error.stderr:
            print(error.stderr)
        return False
    print(f"{name}: exit={completed.returncode}")
    if completed.stdout:
        print(completed.stdout.rstrip())
    if completed.stderr:
        print(completed.stderr.rstrip())
    return completed.returncode == 0


if __name__ == "__main__":
    successes = [run_probe(name, source) for name, source in PROBES.items()]
    raise SystemExit(0 if all(successes) else 1)
