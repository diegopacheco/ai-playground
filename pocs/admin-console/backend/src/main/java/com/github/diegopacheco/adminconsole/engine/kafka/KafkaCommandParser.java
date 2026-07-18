package com.github.diegopacheco.adminconsole.engine.kafka;

import com.github.diegopacheco.adminconsole.engine.redis.RedisCommandParser;
import java.util.List;
import org.springframework.stereotype.Component;

@Component
public class KafkaCommandParser {
    private final RedisCommandParser tokenizer;

    public KafkaCommandParser(RedisCommandParser tokenizer) {
        this.tokenizer = tokenizer;
    }

    public String operation(String command) {
        return tokenizer.parse(command).getFirst().toLowerCase();
    }

    public KafkaCommand parse(String command) {
        List<String> parts = tokenizer.parse(command);
        String operation = parts.getFirst().toLowerCase();
        String target = null;
        Integer partition = null;
        String from = "earliest";
        Long offset = null;
        int limit = 0;
        for (int index = 1; index < parts.size(); index++) {
            String part = parts.get(index);
            switch (part) {
                case "--partition" -> partition = Integer.parseInt(next(parts, ++index, "--partition"));
                case "--limit" -> limit = Integer.parseInt(next(parts, ++index, "--limit"));
                case "--from" -> {
                    from = next(parts, ++index, "--from");
                    if (from.equals("offset")) {
                        offset = Long.parseLong(next(parts, ++index, "--from offset"));
                    }
                }
                default -> {
                    if (part.startsWith("--")) {
                        throw new IllegalArgumentException("unsupported flag: " + part);
                    }
                    if (target == null) {
                        target = part;
                    } else if (operation.equals("list") || operation.equals("describe")) {
                        target = target + " " + part;
                    } else {
                        throw new IllegalArgumentException("too many arguments");
                    }
                }
            }
        }
        return new KafkaCommand(operation, target, partition, from, offset, limit);
    }

    private String next(List<String> parts, int index, String flag) {
        if (index >= parts.size()) {
            throw new IllegalArgumentException(flag + " needs a value");
        }
        return parts.get(index);
    }
}
