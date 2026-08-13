from dataclasses import dataclass


@dataclass(frozen=True)
class Node:
    name: str
    instruction: str


class Graph:
    def __init__(self, nodes, edges, llm):
        self.nodes = nodes
        self.edges = edges
        self.llm = llm
        names = [node.name for node in nodes]
        if len(names) != len(set(names)):
            raise ValueError("Node names must be unique")
        if any(source not in names or target not in names for source, target in edges):
            raise ValueError("Every edge must connect known nodes")

    def run(self, question):
        if not question.strip():
            raise ValueError("question is required")
        pending = list(self.nodes)
        results = {}
        while pending:
            ready = [node for node in pending if all(source in results for source, target in self.edges if target == node.name)]
            if not ready:
                raise ValueError("Graph contains a cycle")
            for node in ready:
                sources = [source for source, target in self.edges if target == node.name]
                context = "\n\n".join(f"{source}:\n{results[source]}" for source in sources)
                prompt = f"{node.instruction}\n\nQuestion:\n{question}"
                if context:
                    prompt = f"{prompt}\n\nInputs:\n{context}"
                results[node.name] = self.llm(prompt)
                pending.remove(node)
        return results


def build_graph(llm):
    nodes = [
        Node("facts", "Identify the essential facts. Be concise."),
        Node("risks", "Identify assumptions and risks. Be concise."),
        Node("answer", "Create the final answer from the supplied inputs. Be concise."),
    ]
    return Graph(nodes, [("facts", "answer"), ("risks", "answer")], llm)
