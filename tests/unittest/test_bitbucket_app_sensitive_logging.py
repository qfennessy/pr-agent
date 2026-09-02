import base64
import json
from types import SimpleNamespace

import pytest
from starlette.background import BackgroundTasks

from pr_agent.servers import bitbucket_app


class _Request:
    def __init__(self, headers, payload):
        self.headers = headers
        self._payload = payload
        self.json_calls = 0

    async def json(self):
        self.json_calls += 1
        return self._payload


class _RecordingLogger:
    def __init__(self):
        self.calls = []

    def __getattr__(self, level):
        def record(*args, **kwargs):
            self.calls.append((level, args, kwargs))

        return record


def _route_endpoint(path, method):
    return next(
        route.endpoint for route in bitbucket_app.router.routes if route.path == path and method in route.methods
    )


async def test_webhook_does_not_log_authorization_header(monkeypatch):
    token = "webhook-authorization-sentinel"
    authorization = f"jWt {token}"
    logger = _RecordingLogger()
    background_tasks = BackgroundTasks()
    monkeypatch.setattr(bitbucket_app, "get_logger", lambda: logger)

    result = await _route_endpoint("/webhook", "POST")(
        background_tasks,
        _Request({"authorization": authorization}, {"event": "pullrequest:created", "data": {}}),
    )

    assert result == "OK"
    assert len(background_tasks.tasks) == 1
    assert token not in repr(logger.calls)


@pytest.mark.parametrize("headers", [{}, {"authorization": "JWT"}, {"authorization": "Bearer token"}])
async def test_webhook_rejects_malformed_authorization_header(monkeypatch, headers):
    logger = _RecordingLogger()
    background_tasks = BackgroundTasks()
    request = _Request(headers, {"event": "pullrequest:created", "data": {}})
    monkeypatch.setattr(bitbucket_app, "get_logger", lambda: logger)

    result = await _route_endpoint("/webhook", "POST")(
        background_tasks,
        request,
    )

    assert result == "OK"
    assert request.json_calls == 0
    assert not background_tasks.tasks
    assert "Bitbucket webhook authorization header is malformed" in repr(logger.calls)


async def test_webhook_validates_connect_jwt_against_installation_client_key(monkeypatch):
    client_key = "installed-client-key"
    claims = base64.urlsafe_b64encode(json.dumps({"iss": client_key, "aud": client_key}).encode()).decode()
    token = f"header.{claims.rstrip('=')}.signature"
    decoded = []
    request_context = {}
    background_tasks = BackgroundTasks()

    def decode(input_jwt, shared_secret, audience, algorithms):
        decoded.append((input_jwt, shared_secret, audience, algorithms))
        return {"iss": client_key, "aud": client_key}

    async def get_bearer_token(shared_secret, key):
        assert (shared_secret, key) == ("shared-secret", client_key)
        return "bearer-token"

    secret_provider = SimpleNamespace(
        get_secret=lambda key: json.dumps({"shared_secret": "shared-secret", "client_key": key})
    )
    monkeypatch.setattr(bitbucket_app.jwt, "decode", decode)
    monkeypatch.setattr(bitbucket_app, "get_bearer_token", get_bearer_token)
    monkeypatch.setattr(bitbucket_app, "get_fork_safe_secret_provider", lambda: secret_provider)
    monkeypatch.setattr(bitbucket_app, "get_settings", lambda: SimpleNamespace(get=lambda *args: "test-app"))
    monkeypatch.setattr(bitbucket_app, "context", request_context)
    monkeypatch.setattr(bitbucket_app, "PRAgent", SimpleNamespace)

    result = await _route_endpoint("/webhook", "POST")(
        background_tasks,
        _Request(
            {"authorization": f"JWT {token}"},
            {"event": "repo:push", "data": {"actor": {"type": "user", "account_id": "actor"}}},
        ),
    )
    await background_tasks()

    assert result == "OK"
    assert decoded == [(token, "shared-secret", client_key, ["HS256"])]
    assert request_context["bitbucket_bearer_token"] == "bearer-token"


async def test_installed_webhook_does_not_log_credentials(monkeypatch):
    authorization = "JWT install-authorization-sentinel"
    shared_secret = "shared-secret-sentinel"
    logger = _RecordingLogger()
    stored = []
    secret_provider = type("SecretProvider", (), {"store_secret": lambda self, *args: stored.append(args)})()
    monkeypatch.setattr(bitbucket_app, "get_logger", lambda: logger)
    monkeypatch.setattr(bitbucket_app, "get_fork_safe_secret_provider", lambda: secret_provider)

    result = await _route_endpoint("/installed", "POST")(
        _Request(
            {"authorization": authorization},
            {"sharedSecret": shared_secret, "clientKey": "client-key", "principal": {"username": "user"}},
        ),
        None,
    )

    logged = repr(logger.calls)
    assert result is None
    assert authorization not in logged
    assert shared_secret not in logged
    assert "handle_installed_webhooks" in logged
    assert json.loads(stored[0][1])["shared_secret"] == shared_secret
