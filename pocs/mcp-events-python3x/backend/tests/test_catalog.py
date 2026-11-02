import pytest

from mcp_events.catalog import ALERT_RAISED, DEPLOY_STATUS_CHANGED, default_catalog
from mcp_events.errors import INVALID_PARAMS, NOT_FOUND, RpcError


def test_deploy_filter_matches_only_the_requested_service_and_env() -> None:
    predicate = default_catalog().get(DEPLOY_STATUS_CHANGED).subscribe({"service": "checkout", "env": "prod"})
    assert predicate({"service": "checkout", "env": "prod", "status": "queued"})
    assert not predicate({"service": "checkout", "env": "dev", "status": "queued"})
    assert not predicate({"service": "search", "env": "prod", "status": "queued"})


def test_alert_min_severity_keeps_more_urgent_alerts_and_drops_less_urgent() -> None:
    predicate = default_catalog().get(ALERT_RAISED).subscribe({"minSeverity": "P2"})
    assert predicate({"severity": "P1"})
    assert predicate({"severity": "P2"})
    assert not predicate({"severity": "P3"})


def test_arguments_outside_the_input_schema_are_rejected_as_invalid_params() -> None:
    event_type = default_catalog().get(DEPLOY_STATUS_CHANGED)
    with pytest.raises(RpcError) as unknown:
        event_type.subscribe({"region": "us"})
    with pytest.raises(RpcError) as bad_enum:
        event_type.subscribe({"env": "qa"})
    assert unknown.value.code == INVALID_PARAMS
    assert bad_enum.value.code == INVALID_PARAMS


def test_unknown_event_type_is_not_found() -> None:
    with pytest.raises(RpcError) as error:
        default_catalog().get("email.received")
    assert error.value.code == NOT_FOUND
