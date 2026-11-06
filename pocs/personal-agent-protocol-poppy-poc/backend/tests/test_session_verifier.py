import pytest

from poppy_company.agent_directory import AgentDirectory
from poppy_company.errors import PoppyError
from poppy_company.session_verifier import SessionVerifier
from signer import BASE, SigningAgent

URL = f"{BASE}/api/rooms"


def verifier_for(agent: SigningAgent) -> SessionVerifier:
    directory = AgentDirectory(BASE)
    directory.publish(agent.name, agent.metadata())
    return SessionVerifier(directory, BASE, lambda: agent.now)


def rejects(verifier: SessionVerifier, token: str, method: str = "GET", url: str = URL) -> str:
    with pytest.raises(PoppyError) as raised:
        verifier.verify(token, method, url)
    assert raised.value.code == "invalid_session"
    return raised.value.description


def test_valid_session_yields_only_the_pairwise_identity() -> None:
    agent = SigningAgent()
    session = verifier_for(agent).verify(agent.session("GET", URL), "GET", URL)
    assert (session.client_id, session.sub, session.sid) == (agent.client_id, "usr_alpha", "sid_one")


def test_tampered_signature_is_rejected_so_a_proxy_cannot_rewrite_claims() -> None:
    agent = SigningAgent()
    token = agent.session("GET", URL)
    head, payload, signature = token.split(".")
    flipped = ("A" if signature[0] != "A" else "B") + signature[1:]
    assert "signature" in rejects(verifier_for(agent), f"{head}.{payload}.{flipped}")


def test_token_bound_to_one_request_cannot_be_reused_for_another_url_or_method() -> None:
    agent = SigningAgent()
    verifier = verifier_for(agent)
    assert "htu" in rejects(verifier, agent.session("GET", URL), "GET", f"{BASE}/api/holds")
    assert "htm" in rejects(verifier, agent.session("GET", URL), "POST", URL)


def test_replayed_jti_is_rejected() -> None:
    agent = SigningAgent()
    verifier = verifier_for(agent)
    token = agent.session("GET", URL)
    verifier.verify(token, "GET", URL)
    assert "jti" in rejects(verifier, token)


def test_token_for_another_company_is_rejected() -> None:
    agent = SigningAgent()
    assert "aud" in rejects(verifier_for(agent), agent.session("GET", URL, aud="https://other.example"))


def test_stale_iat_is_rejected() -> None:
    agent = SigningAgent()
    assert "iat" in rejects(verifier_for(agent), agent.session("GET", URL, iat=agent.now - 600))


def test_expired_delegation_is_rejected_so_leaked_delegated_keys_age_out() -> None:
    agent = SigningAgent()
    assert "expired" in rejects(verifier_for(agent), agent.session("GET", URL, delegation=agent.delegation(exp_offset=-1)))


def test_delegation_signed_by_a_key_the_agent_never_published_is_rejected() -> None:
    agent = SigningAgent()
    impostor = SigningAgent().root
    assert "signature" in rejects(verifier_for(agent), agent.session("GET", URL, delegation=agent.delegation(signer=impostor)))


def test_unknown_personal_agent_is_rejected() -> None:
    agent = SigningAgent()
    stranger = SigningAgent(name="stranger")
    assert "unknown" in rejects(verifier_for(agent), stranger.session("GET", URL, iss=stranger.client_id))
