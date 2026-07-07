from app import build_agent


def ids(result):
    return {item["id"] for item in result["decisions"] if item["allowed"]}


def test_beth_gets_engineering_context():
    result = build_agent().answer("user:beth", "token roadmap approval architecture")
    assert result["delegated"]
    assert "document:api_design" in ids(result)
    assert "document:architecture" in ids(result)
    assert "document:roadmap" in ids(result)
    assert "document:payroll" not in ids(result)


def test_carl_gets_only_direct_roadmap_access():
    result = build_agent().answer("user:carl", "roadmap architecture")
    assert result["delegated"]
    assert ids(result) == {"document:roadmap"}


def test_dana_stops_without_delegation():
    result = build_agent().answer("user:dana", "payroll")
    assert not result["delegated"]
    assert result["decisions"] == []


def test_owner_inherits_folder_view():
    result = build_agent().answer("user:beth", "payroll")
    assert result["delegated"]
    assert ids(result) == set()


if __name__ == "__main__":
    test_beth_gets_engineering_context()
    test_carl_gets_only_direct_roadmap_access()
    test_dana_stops_without_delegation()
    test_owner_inherits_folder_view()
    print("tests passed")
