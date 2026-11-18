import asyncio
import time

import pytest

from agent_service import activities
from agent_service.activities import with_heartbeat
from agent_service.config import temporal_address, temporal_ui_url


def test_a_long_agent_call_keeps_heartbeating_so_a_dead_worker_is_noticed_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(activities, "HEARTBEAT_SECONDS", 0.05)
    beats: list[float] = []

    def slow() -> str:
        time.sleep(0.3)
        return "done"

    assert asyncio.run(with_heartbeat(slow, lambda: beats.append(time.monotonic()))) == "done"
    assert len(beats) >= 3


def test_a_failing_agent_call_fails_the_activity_so_temporal_retries_it() -> None:
    def broken() -> str:
        raise RuntimeError("cli exploded")

    with pytest.raises(RuntimeError, match="cli exploded"):
        asyncio.run(with_heartbeat(broken, lambda: None))


def test_temporal_settings_fail_loud_when_missing() -> None:
    assert temporal_address({"TEMPORAL_ADDRESS": "localhost:7233"}) == "localhost:7233"
    assert temporal_ui_url({"TEMPORAL_UI_URL": "http://localhost:8233/"}) == "http://localhost:8233"
    with pytest.raises(ValueError, match="TEMPORAL_ADDRESS"):
        temporal_address({})
    with pytest.raises(ValueError, match="TEMPORAL_UI_URL"):
        temporal_ui_url({"TEMPORAL_UI_URL": "localhost"})
