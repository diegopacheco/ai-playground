from asgi import new_app, obj, objs, signed
from poppy_company.company_agent import CompanyAgent
from poppy_company.rooms import RoomsService
from poppy_company.json_types import JsonObject
from signer import SigningAgent


def first(text: str, message_id: str = "msg_1") -> JsonObject:
    return {"message": {"id": message_id, "author": {"type": "agent"}, "text": text}}


def test_company_agent_answers_and_points_unsupported_tasks_to_feedback() -> None:
    agent = CompanyAgent(RoomsService())
    assert "3 pm" in agent.reply("What time is check-in?")
    assert "feedback" in agent.reply("Can I pay with a card?")
    assert "River Suite" in agent.reply("Which rooms are available?")


def test_conversation_flow_with_cursor_reads_only_new_events() -> None:
    agent = SigningAgent()
    app = new_app(agent)
    started = signed(app, agent, "POST", "/poppy/conversations", first("Do you allow dogs?"))
    assert (started.status, started.json()["responder"]) == (201, "agent")
    cid = started.json()["conversation_id"]
    read = signed(app, agent, "GET", f"/poppy/conversations/{cid}/events").json()
    messages = [obj(event["message"]) for event in objs(read["events"])]
    assert [message["role"] for message in messages] == ["user", "company"]
    assert "Pets are welcome" in str(messages[1]["text"])
    signed(app, agent, "POST", f"/poppy/conversations/{cid}/messages", {"message": {"id": "msg_2", "text": "Is parking free?"}})
    newer = objs(signed(app, agent, "GET", f"/poppy/conversations/{cid}/events?cursor={read['cursor']}").json()["events"])
    assert len(newer) == 2 and "Parking is free" in str(obj(newer[1]["message"])["text"])


def test_other_user_cannot_see_the_conversation() -> None:
    alpha = SigningAgent()
    app = new_app(alpha)
    cid = signed(app, alpha, "POST", "/poppy/conversations", first("hello")).json()["conversation_id"]
    beta = SigningAgent(sub="usr_beta", root=alpha.root, delegated=alpha.delegated)
    response = signed(app, beta, "GET", f"/poppy/conversations/{cid}/events")
    assert (response.status, response.json()["error"]) == (404, "conversation_not_found")


def test_resent_message_id_is_idempotent_but_changed_content_conflicts() -> None:
    agent = SigningAgent()
    app = new_app(agent)
    cid = signed(app, agent, "POST", "/poppy/conversations", first("hello")).json()["conversation_id"]
    assert signed(app, agent, "POST", f"/poppy/conversations/{cid}/messages", first("hello")).status == 202
    assert len(objs(signed(app, agent, "GET", f"/poppy/conversations/{cid}/events").json()["events"])) == 2
    conflict = signed(app, agent, "POST", f"/poppy/conversations/{cid}/messages", first("changed"))
    assert (conflict.status, conflict.json()["error"]) == (409, "message_id_conflict")


def test_closed_conversation_takes_no_more_messages() -> None:
    agent = SigningAgent()
    app = new_app(agent)
    cid = signed(app, agent, "POST", "/poppy/conversations", first("hello")).json()["conversation_id"]
    assert signed(app, agent, "POST", f"/poppy/conversations/{cid}/close").json()["status"] == "closed"
    closed = signed(app, agent, "POST", f"/poppy/conversations/{cid}/messages", first("again", "msg_9"))
    assert (closed.status, closed.json()["error"]) == (409, "conversation_closed")
