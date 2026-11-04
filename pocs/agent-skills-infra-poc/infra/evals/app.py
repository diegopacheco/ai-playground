import re

from common.http import Authenticator, HttpError, Request, Response, Router, asgi_app, health_handler, json_response
from common.secrets import load_signing_key
from common.settings import Settings, load_settings

from evals.graders import EVALUATORS
from evals.store import EvalStore, grade_cases

SUITE_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,100}$")
READ_ALL = "evals:read:all"


def build_router(settings: Settings) -> Router:
    auth = Authenticator(load_signing_key(settings.secrets_dir), settings.issuer)
    store = EvalStore(settings.database_url)

    async def evaluators(request: Request) -> Response:
        auth.principal(request)
        return json_response(
            {"items": [{"name": e.name, "description": e.description, "value": e.value_hint} for e in EVALUATORS.values()]}
        )

    async def create_run(request: Request) -> Response:
        claims = auth.require(request, "evals:write")
        body = request.json()
        if not isinstance(body, dict):
            raise HttpError(400, "invalid_request", "body must be an object")
        suite = str(body.get("suite", ""))
        if not SUITE_PATTERN.match(suite):
            raise HttpError(400, "invalid_suite", "suite must match [A-Za-z0-9._-]{1,100}")
        cases = grade_cases(body.get("cases"))
        run = await store.save(claims.sub, suite, cases)
        run["cases"] = [
            {"name": c.name, "passed": c.passed, "latency_ms": c.latency_ms, "checks": c.checks} for c in cases
        ]
        return json_response(run, 201)

    async def list_runs(request: Request) -> Response:
        claims = auth.require(request, "evals:read", READ_ALL)
        try:
            limit = max(1, min(500, int(request.query.get("limit", "100"))))
        except ValueError as error:
            raise HttpError(400, "invalid_request", "limit must be an integer") from error
        runs = await store.list_runs(
            claims.sub, READ_ALL in claims.scopes, request.query.get("suite"), limit, request.query.get("tenant") or None
        )
        return json_response({"items": runs})

    async def get_run(request: Request) -> Response:
        claims = auth.require(request, "evals:read", READ_ALL)
        if not request.params["id"].isdigit():
            raise HttpError(400, "invalid_request", "id must be an integer")
        run = await store.get_run(claims.sub, READ_ALL in claims.scopes, int(request.params["id"]))
        if run is None:
            raise HttpError(404, "not_found", "run not found")
        return json_response(run)

    router = Router()
    router.add("GET", "/health", health_handler("evals"))
    router.add("GET", "/v1/evals/evaluators", evaluators)
    router.add("POST", "/v1/evals/runs", create_run)
    router.add("GET", "/v1/evals/runs", list_runs)
    router.add("GET", "/v1/evals/runs/{id}", get_run)
    return router


app = asgi_app(build_router(load_settings()))
