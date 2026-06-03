def test_health_returns_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "Chatbot Service"
    assert "version" in body


def test_ready_returns_true_in_mock_mode(client):
    r = client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is True
    assert body["backend"] == "MockBackend"
    assert body["model_path"] is None


def test_model_info_reflects_mock_mode(client):
    r = client.get("/model/info")
    assert r.status_code == 200
    body = r.json()
    assert body["backend_type"] == "MockBackend"
    assert body["model_mode"] == "mock"
    assert body["model_path"] is None
    assert body["base_model"] is None
    assert body["max_new_tokens"] == 512
    assert body["temperature"] == 0.3
