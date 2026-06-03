def test_chat_qa_happy_path(client):
    r = client.post("/chat", json={"message": "What is DOGE?", "mode": "qa"})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "qa"
    assert body["answer"] is not None
    assert body["distill_result"] is None
    assert body["latency_ms"] >= 0


def test_chat_distill_happy_path(client):
    r = client.post(
        "/chat",
        json={
            "message": "What happened with ICE raids last week?",
            "mode": "distill",
            "today": "2026-06-03",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "distill"
    assert body["distill_result"] is not None
    assert body["answer"] is None
    dr = body["distill_result"]
    assert "intent" in dr
    assert "time_type" in dr
    assert "needs_context" in dr
    assert "query" in dr


def test_chat_distill_with_context(client):
    r = client.post(
        "/chat",
        json={
            "message": "Summarize this",
            "mode": "distill",
            "context": "Some background context here.",
        },
    )
    assert r.status_code == 200


def test_chat_empty_message_is_rejected(client):
    r = client.post("/chat", json={"message": "", "mode": "qa"})
    assert r.status_code == 422


def test_chat_invalid_mode_is_rejected(client):
    r = client.post("/chat", json={"message": "hello", "mode": "invalid"})
    assert r.status_code == 422


def test_chat_context_too_long_is_rejected(client):
    r = client.post(
        "/chat",
        json={"message": "hello", "mode": "distill", "context": "x" * 2001},
    )
    assert r.status_code == 422


def test_chat_message_too_long_is_rejected(client):
    r = client.post("/chat", json={"message": "x" * 4097, "mode": "qa"})
    assert r.status_code == 422


def test_response_includes_request_id_header(client):
    r = client.post("/chat", json={"message": "hello", "mode": "qa"})
    assert "x-request-id" in r.headers


def test_custom_request_id_is_echoed(client):
    r = client.post(
        "/chat",
        json={"message": "hello", "mode": "qa"},
        headers={"X-Request-ID": "test-id-123"},
    )
    assert r.headers["x-request-id"] == "test-id-123"
