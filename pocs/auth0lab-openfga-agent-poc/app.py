from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import re


ROOT = Path(__file__).parent


@dataclass(frozen=True)
class TupleKey:
    user: str
    relation: str
    object: str


class LocalFga:
    def __init__(self, tuples):
        self.tuples = {TupleKey(**item) for item in tuples}

    def check(self, user, relation, object):
        if self.direct(user, relation, object):
            return True
        object_type = object.split(":", 1)[0]
        if object_type == "folder":
            return self.check_folder(user, relation, object)
        if object_type == "document":
            return self.check_document(user, relation, object)
        return False

    def direct(self, user, relation, object):
        return TupleKey(user, relation, object) in self.tuples

    def check_folder(self, user, relation, object):
        if relation == "viewer":
            return self.check(user, "owner", object)
        return False

    def check_document(self, user, relation, object):
        if relation == "editor":
            return self.check(user, "owner", object)
        if relation == "viewer":
            return self.check(user, "owner", object) or self.inherited_folder_view(user, object)
        if relation in {"can_cite", "can_summarize"}:
            return self.check(user, "viewer", object)
        if relation == "can_write":
            return self.check(user, "editor", object)
        return False

    def inherited_folder_view(self, user, document):
        for item in self.tuples:
            if item.object == document and item.relation == "parent":
                if self.check(user, "viewer", item.user):
                    return True
        return False


class IdentityAgent:
    def __init__(self, agent_id, fga, documents):
        self.agent_id = agent_id
        self.fga = fga
        self.documents = documents

    def answer(self, user, query):
        delegated = self.fga.check(user, "delegate", self.agent_id)
        if not delegated:
            return {
                "user": user,
                "agent": self.agent_id,
                "query": query,
                "delegated": False,
                "decisions": [],
                "answer": "Authorization stopped before retrieval.",
            }
        candidates = self.retrieve(query)
        decisions = []
        allowed = []
        for document in candidates:
            can_cite = self.fga.check(user, "can_cite", document["id"])
            decisions.append({"id": document["id"], "title": document["title"], "allowed": can_cite})
            if can_cite:
                allowed.append(document)
        return {
            "user": user,
            "agent": self.agent_id,
            "query": query,
            "delegated": True,
            "decisions": decisions,
            "answer": self.compose_answer(allowed),
        }

    def retrieve(self, query):
        query_terms = terms(query)
        ranked = []
        for document in self.documents:
            haystack = terms(document["title"] + " " + document["text"])
            score = sum(1 for term in query_terms if term in haystack)
            if score:
                ranked.append((score, document["id"], document))
        ranked.sort(reverse=True)
        return [document for _, _, document in ranked]

    def compose_answer(self, documents):
        if not documents:
            return "No authorized context matched the query."
        lines = []
        for document in documents:
            lines.append(f"{document['title']}: {document['text']} [{document['id']}]")
        return "\n".join(lines)


def terms(value):
    return set(re.findall(r"[a-z0-9_]+", value.lower()))


def load_json(name):
    return json.loads((ROOT / name).read_text())


def render(result):
    print(f"Principal: {result['user']}")
    print(f"Agent: {result['agent']}")
    print(f"Query: {result['query']}")
    print(f"Delegation: {'allowed' if result['delegated'] else 'denied'}")
    if result["decisions"]:
        print("Candidate checks:")
        for item in result["decisions"]:
            decision = "allowed" if item["allowed"] else "denied"
            print(f"  {item['id']} {decision} {item['title']}")
    print("Answer:")
    print(result["answer"])


def build_agent():
    fga = LocalFga(load_json("tuples.json"))
    return IdentityAgent("agent:secprep", fga, load_json("documents.json"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="user:beth")
    parser.add_argument("--query", default="token roadmap approval")
    args = parser.parse_args()
    render(build_agent().answer(args.user, args.query))


if __name__ == "__main__":
    main()
