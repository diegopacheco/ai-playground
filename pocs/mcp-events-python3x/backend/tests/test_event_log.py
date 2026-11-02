from datetime import UTC, datetime, timedelta

from mcp_events.event_log import EventLog


class FakeClock:
    def __init__(self) -> None:
        self.now = datetime(2026, 11, 2, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


def accept_all(_event: object) -> bool:
    return True


def test_read_returns_only_events_after_the_cursor_so_clients_never_see_an_event_twice() -> None:
    log = EventLog()
    log.append("a", {"n": 1})
    log.append("a", {"n": 2})
    read = log.read(1, accept_all)
    assert [e.data["n"] for e in read.events] == [2]
    assert read.position == 2
    assert not read.truncated


def test_cursor_advances_past_filtered_out_events_so_a_quiet_subscription_does_not_go_stale() -> None:
    log = EventLog()
    log.append("other", {})
    log.append("other", {})
    read = log.read(0, lambda e: e.name == "wanted")
    assert read.events == []
    assert read.position == 2


def test_limit_sets_has_more_and_a_mid_cursor_so_the_backlog_drains_in_order() -> None:
    log = EventLog()
    for n in range(5):
        log.append("a", {"n": n})
    first = log.read(0, accept_all, limit=2)
    assert [e.data["n"] for e in first.events] == [0, 1]
    assert first.has_more
    second = log.read(first.position, accept_all, limit=10)
    assert [e.data["n"] for e in second.events] == [2, 3, 4]
    assert not second.has_more


def test_evicted_events_mark_truncated_so_the_client_knows_it_missed_something() -> None:
    log = EventLog(capacity=2)
    for n in range(5):
        log.append("a", {"n": n})
    read = log.read(1, accept_all)
    assert read.truncated
    assert [e.data["n"] for e in read.events] == [3, 4]


def test_max_age_floor_skips_old_events_and_reports_truncated() -> None:
    clock = FakeClock()
    log = EventLog(clock=clock)
    log.append("a", {"n": "old"})
    clock.now += timedelta(minutes=10)
    log.append("a", {"n": "new"})
    read = log.read(0, accept_all, not_before=clock.now - timedelta(minutes=1))
    assert [e.data["n"] for e in read.events] == ["new"]
    assert read.truncated


def test_cursor_from_a_previous_server_run_is_truncated_to_now_instead_of_failing() -> None:
    log = EventLog()
    log.append("a", {})
    read = log.read(99, accept_all)
    assert read.truncated
    assert read.position == 1
