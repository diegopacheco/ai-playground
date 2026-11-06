from asgi import new_app, objs, request, signed
from signer import SigningAgent


def test_discovery_needs_no_session_and_is_valid_for_the_inspector() -> None:
    agent = SigningAgent()
    app = new_app(agent)
    document = request(app, "GET", "/.well-known/poppy.json").json()
    assert document["protocol_version"] == "0.4"
    report = request(app, "POST", "/inspector/validate", {"document": document}).json()
    assert report["valid"] is True


def test_resource_without_session_is_rejected_and_without_version_is_unsupported() -> None:
    app = new_app(SigningAgent())
    assert request(app, "GET", "/api/rooms", headers={"poppy-version": "0.4"}).json()["error"] == "invalid_session"
    response = request(app, "GET", "/api/rooms")
    assert (response.status, response.json()["error"]) == (400, "version_unsupported")


def test_room_search_filters_by_price_and_pets() -> None:
    agent = SigningAgent()
    rooms = objs(signed(new_app(agent), agent, "GET", "/api/rooms?max_price=200&pets=true").json()["rooms"])
    assert [room["id"] for room in rooms] == ["rm_oak"]


def test_holds_follow_the_pairwise_user_id_and_never_leak_across_users() -> None:
    alpha = SigningAgent()
    app = new_app(alpha)
    beta = SigningAgent(sub="usr_beta", root=alpha.root, delegated=alpha.delegated)
    created = signed(app, alpha, "POST", "/api/holds", {"room_id": "rm_river", "nights": 2})
    assert (created.status, created.json()["total"]) == (201, 640)
    assert [hold["room_id"] for hold in objs(signed(app, alpha, "GET", "/api/holds").json()["holds"])] == ["rm_river"]
    assert signed(app, beta, "GET", "/api/holds").json()["holds"] == []


def test_hold_rejects_out_of_range_nights() -> None:
    agent = SigningAgent()
    assert signed(new_app(agent), agent, "POST", "/api/holds", {"room_id": "rm_oak", "nights": 30}).status == 400


def test_feedback_accepts_a_task_and_rejects_an_oversized_one() -> None:
    agent = SigningAgent()
    app = new_app(agent)
    assert signed(app, agent, "POST", "/poppy/feedback", {"task": "Pay with a gift card", "description": "No gift card field."}).status == 202
    assert signed(app, agent, "POST", "/poppy/feedback", {"task": "x" * 201, "description": "too long"}).status == 400
    assert [report.task for report in app.feedback.reports] == ["Pay with a gift card"]


def test_cors_allows_only_local_origins() -> None:
    app = new_app(SigningAgent())
    local = request(app, "OPTIONS", "/api/rooms", headers={"origin": "http://localhost:5341"})
    remote = request(app, "OPTIONS", "/api/rooms", headers={"origin": "https://evil.example"})
    assert "poppy-session" in local.headers["access-control-allow-headers"]
    assert "access-control-allow-origin" not in remote.headers
