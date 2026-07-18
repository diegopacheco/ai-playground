package com.github.diegopacheco.adminconsole.ai;

import java.util.ArrayList;
import java.util.List;
import org.springframework.stereotype.Component;

@Component
public class SuggestionParser {
    public String extract(String output) {
        if (output == null || output.isBlank()) {
            throw new IllegalStateException("the agent returned nothing");
        }
        List<String> fenced = fencedBlocks(output);
        if (!fenced.isEmpty()) {
            return fenced.getFirst().trim();
        }
        List<String> kept = new ArrayList<>();
        for (String line : output.strip().lines().toList()) {
            String trimmed = line.strip();
            if (trimmed.isEmpty() && kept.isEmpty()) {
                continue;
            }
            kept.add(line);
        }
        return String.join("\n", kept).trim();
    }

    private List<String> fencedBlocks(String output) {
        List<String> blocks = new ArrayList<>();
        String[] lines = output.split("\n", -1);
        StringBuilder current = null;
        for (String line : lines) {
            String trimmed = line.strip();
            if (trimmed.startsWith("```")) {
                if (current == null) {
                    current = new StringBuilder();
                } else {
                    blocks.add(current.toString());
                    current = null;
                }
                continue;
            }
            if (current != null) {
                current.append(line).append('\n');
            }
        }
        if (current != null && !current.isEmpty()) {
            blocks.add(current.toString());
        }
        return blocks;
    }
}
