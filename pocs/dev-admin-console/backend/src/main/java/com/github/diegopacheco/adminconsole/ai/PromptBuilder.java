package com.github.diegopacheco.adminconsole.ai;

import com.github.diegopacheco.adminconsole.engine.SchemaNode;
import com.github.diegopacheco.adminconsole.project.ConnectionConfig;
import com.github.diegopacheco.adminconsole.project.ConnectionKind;
import java.util.List;
import org.springframework.stereotype.Component;

@Component
public class PromptBuilder {
    private static final int MAX_OBJECTS = 60;
    private static final int MAX_CHILDREN = 40;

    public String build(ConnectionConfig config, List<SchemaNode> schema, String request) {
        if (request == null || request.isBlank()) {
            throw new IllegalArgumentException("describe what you want the query to do");
        }
        StringBuilder prompt = new StringBuilder();
        prompt.append("You write read-only queries for an admin console.\n\n");
        prompt.append("Target: ").append(grammar(config.kind())).append("\n\n");
        prompt.append("Rules:\n");
        prompt.append("- The query must only read. Never write, update, delete, drop or alter anything.\n");
        prompt.append("- Answer with the query only, inside one code fence. No explanation.\n");
        prompt.append("- Use only the names listed below. Do not invent tables, columns or keys.\n\n");
        prompt.append("Available names:\n");
        appendSchema(prompt, schema, 0, MAX_OBJECTS);
        prompt.append("\nRequest: ").append(request.strip()).append('\n');
        return prompt.toString();
    }

    public record FederatedSource(String connectionName, String kind, String source, List<String> columns) {}

    public String buildFederated(List<FederatedSource> sources, String request) {
        if (request == null || request.isBlank()) {
            throw new IllegalArgumentException("describe the join you want");
        }
        StringBuilder prompt = new StringBuilder();
        prompt.append("You write cross-engine join queries for an admin console.\n\n");
        prompt.append("""
                Grammar — exactly this shape, nothing else:
                  SELECT <alias>.<column>, ... FROM <connection>.<source> <alias>
                  JOIN <connection>.<source> <alias> ON <alias>.<column> = <alias>.<column>
                  [JOIN <connection>.<source> <alias> ON <alias>.<column> = <alias>.<column>] ...
                  LIMIT <n>

                Rules:
                - Up to 5 sources may be joined, so up to 4 JOIN clauses. Two sources is the minimum.
                - Sources may come from the same connection or from different ones.
                - Each JOIN carries exactly one equality, and it must compare the source being
                  joined to an alias introduced earlier in the query.
                - INNER or LEFT only. No GROUP BY, no aggregation, no subqueries.
                - Every projected column must be qualified with its alias.
                - Use only the connection names, sources and columns listed below. Never invent one.
                - Join keys must be columns that actually exist on their side and hold comparable values.
                - Answer with the query only, inside one code fence. No explanation.

                Available sources:
                """);
        for (FederatedSource source : sources) {
            prompt.append("- ").append(source.connectionName()).append('.').append(source.source())
                    .append("  [").append(source.kind()).append("]  columns: ")
                    .append(String.join(", ", source.columns())).append('\n');
        }
        prompt.append("\nRequest: ").append(request.strip()).append('\n');
        return prompt.toString();
    }

    String grammar(ConnectionKind kind) {
        return switch (kind) {
            case POSTGRES -> "PostgreSQL. Write standard PostgreSQL SELECT syntax.";
            case MYSQL -> "MySQL. Write MySQL SELECT syntax.";
            case CASSANDRA -> "Apache Cassandra. Write CQL SELECT only. "
                    + "Filter on partition keys; add ALLOW FILTERING only when unavoidable.";
            case REDIS -> "Redis. Write one read-only Redis command, for example GET, HGETALL, LRANGE, SMEMBERS, ZRANGE or SCAN.";
            case ETCD -> "etcd. Write an etcdctl-style read command: get <key>, get <prefix> --prefix, or range.";
            case KAFKA -> "Kafka. Write one console command: list topics, describe topic <t>, offsets <t>, "
                    + "or consume <t> [--partition N] [--from earliest|latest] [--limit N].";
            case ELASTICSEARCH -> "Elasticsearch. Write a Dev Tools style request: "
                    + "GET /<index>/_search followed by a JSON query body. Only GET is allowed.";
        };
    }

    private void appendSchema(StringBuilder prompt, List<SchemaNode> nodes, int depth, int limit) {
        int written = 0;
        for (SchemaNode node : nodes) {
            if (written++ >= limit) {
                prompt.append("  ".repeat(depth)).append("- … more omitted\n");
                return;
            }
            prompt.append("  ".repeat(depth)).append("- ").append(node.name());
            if (node.detail() != null && !node.detail().isBlank() && depth > 0) {
                prompt.append(" (").append(node.detail()).append(')');
            }
            prompt.append('\n');
            if (node.children() != null && !node.children().isEmpty() && depth < 2) {
                appendSchema(prompt, node.children(), depth + 1, MAX_CHILDREN);
            }
        }
    }
}
