from dataclasses import dataclass

CONSUMER_SCOPES = ("state:read", "state:write", "logs:write", "logs:read", "evals:write", "evals:read")
ADMIN_SCOPES = ("clients:read", "state:read:all", "logs:read:all", "evals:read:all")


@dataclass(frozen=True)
class CatalogEntry:
    client_id: str
    kind: str
    description: str
    scopes: tuple[str, ...]


CATALOG: tuple[CatalogEntry, ...] = (
    CatalogEntry("decision-log", "skill", "Architecture decisions kept across sessions", CONSUMER_SCOPES),
    CatalogEntry("answer-grader", "skill", "Grades answers with the shared evaluators", CONSUMER_SCOPES),
    CatalogEntry("bookmarks-mcp", "mcp", "Bookmarks with tags and search", CONSUMER_SCOPES),
    CatalogEntry("feature-flags-mcp", "mcp", "Feature flags with rollout buckets and audit", CONSUMER_SCOPES),
    CatalogEntry("infra-admin", "admin", "Console: reads every tenant, writes nothing", ADMIN_SCOPES),
    CatalogEntry("eval-viewer", "admin", "Evals UI: reads every eval run", ("evals:read:all",)),
    CatalogEntry("itest-alpha", "test", "Integration tests, tenant A", CONSUMER_SCOPES),
    CatalogEntry("itest-beta", "test", "Integration tests, tenant B", CONSUMER_SCOPES),
    CatalogEntry("itest-readonly", "test", "Integration tests, read only state", ("state:read",)),
)

BY_ID: dict[str, CatalogEntry] = {entry.client_id: entry for entry in CATALOG}
