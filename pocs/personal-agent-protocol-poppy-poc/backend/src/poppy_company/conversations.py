from dataclasses import dataclass, field

from poppy_company.company_agent import CompanyAgent
from poppy_company.errors import PoppyError, invalid_request
from poppy_company.json_types import JsonObject, JsonValue
from poppy_company.session_verifier import Session

MAX_TEXT = 4000


@dataclass
class Conversation:
    id: str
    client_id: str
    sub: str
    status: str = "idle"
    events: list[JsonObject] = field(default_factory=list)
    message_texts: dict[str, str] = field(default_factory=dict)

    def owned_by(self, session: Session) -> bool:
        return self.client_id == session.client_id and self.sub == session.sub

    def summary(self) -> JsonObject:
        return {"conversation_id": self.id, "status": self.status, "responder": "agent"}


def parse_message(body: JsonObject) -> tuple[str, str]:
    message = body.get("message")
    if not isinstance(message, dict):
        raise invalid_request("message is required")
    message_id, text = message.get("id"), message.get("text")
    if not isinstance(message_id, str) or not message_id.startswith("msg_"):
        raise invalid_request("message.id must be a string starting with msg_")
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT:
        raise invalid_request(f"message.text must be 1 to {MAX_TEXT} characters")
    return message_id, text


class Conversations:
    def __init__(self, agent: CompanyAgent) -> None:
        self._agent = agent
        self._items: dict[str, Conversation] = {}
        self._next = 0

    def start(self, session: Session, body: JsonObject) -> JsonObject:
        message_id, text = parse_message(body)
        self._next += 1
        conversation = Conversation(f"cnv_{self._next}", session.client_id, session.sub)
        self._items[conversation.id] = conversation
        self._exchange(conversation, message_id, text)
        return conversation.summary()

    def send(self, session: Session, conversation_id: str, body: JsonObject) -> JsonObject:
        conversation = self._open(session, conversation_id)
        message_id, text = parse_message(body)
        known = conversation.message_texts.get(message_id)
        if known is not None and known != text:
            raise PoppyError(409, "message_id_conflict", "message id was accepted with different content")
        if known is None:
            self._exchange(conversation, message_id, text)
        return conversation.summary()

    def events(self, session: Session, conversation_id: str, cursor: str | None) -> JsonObject:
        conversation = self._owned(session, conversation_id)
        start = 0
        if cursor:
            if not cursor.isdigit() or int(cursor) > len(conversation.events):
                raise PoppyError(400, "invalid_cursor", "the cursor doesn't belong to this conversation")
            start = int(cursor)
        events: list[JsonValue] = list(conversation.events[start:])
        return conversation.summary() | {"events": events, "cursor": str(len(conversation.events))}

    def close(self, session: Session, conversation_id: str) -> JsonObject:
        conversation = self._open(session, conversation_id)
        conversation.status = "closed"
        self._append(conversation, {"type": "state", "status": "closed", "responder": "agent"})
        return conversation.summary()

    def _owned(self, session: Session, conversation_id: str) -> Conversation:
        conversation = self._items.get(conversation_id)
        if conversation is None or not conversation.owned_by(session):
            raise PoppyError(404, "conversation_not_found", "the conversation is unknown")
        return conversation

    def _open(self, session: Session, conversation_id: str) -> Conversation:
        conversation = self._owned(session, conversation_id)
        if conversation.status == "closed":
            raise PoppyError(409, "conversation_closed", "the conversation takes no more messages")
        return conversation

    def _exchange(self, conversation: Conversation, message_id: str, text: str) -> None:
        conversation.message_texts[message_id] = text
        self._append(conversation, {"type": "message", "message": {"id": message_id, "role": "user", "author": {"type": "agent"}, "text": text}})
        reply = self._agent.reply(text)
        self._append(conversation, {"type": "message", "message": {"id": f"msg_c{len(conversation.events)}", "role": "company", "author": {"type": "agent"}, "text": reply}})

    def _append(self, conversation: Conversation, event: JsonObject) -> None:
        conversation.events.append({"id": f"evt_{len(conversation.events) + 1}"} | event)
