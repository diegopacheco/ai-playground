package com.github.diegopacheco.devadminconsole.engine.etcd;

import com.github.diegopacheco.devadminconsole.engine.SchemaNode;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.springframework.stereotype.Component;

@Component
public class PrefixTreeBuilder {
    private static final class Node {
        private final Map<String, Node> children = new LinkedHashMap<>();
        private String value;
        private boolean terminal;
    }

    public List<SchemaNode> build(Map<String, String> keys) {
        Node root = new Node();
        keys.forEach((key, value) -> {
            Node current = root;
            for (String segment : segments(key)) {
                current = current.children.computeIfAbsent(segment, name -> new Node());
            }
            current.terminal = true;
            current.value = value;
        });
        return toNodes(root);
    }

    List<String> segments(String key) {
        List<String> segments = new ArrayList<>();
        for (String segment : key.split("/")) {
            if (!segment.isEmpty()) {
                segments.add(segment);
            }
        }
        if (segments.isEmpty()) {
            segments.add(key);
        }
        return segments;
    }

    private List<SchemaNode> toNodes(Node node) {
        List<SchemaNode> result = new ArrayList<>();
        node.children.forEach((name, child) -> {
            List<SchemaNode> children = toNodes(child);
            String kind = child.terminal ? "key" : "prefix";
            String detail = child.terminal ? child.value : children.size() + " keys";
            result.add(new SchemaNode(name, kind, detail, children));
        });
        return result;
    }
}
