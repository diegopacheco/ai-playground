package com.github.diegopacheco.devadminconsole.engine.etcd;

import com.github.diegopacheco.devadminconsole.engine.redis.RedisCommandParser;
import java.util.List;
import org.springframework.stereotype.Component;

@Component
public class EtcdCommandParser {
    private final RedisCommandParser tokenizer;

    public EtcdCommandParser(RedisCommandParser tokenizer) {
        this.tokenizer = tokenizer;
    }

    public EtcdCommand parse(String command) {
        List<String> parts = tokenizer.parse(command);
        String operation = parts.getFirst().toLowerCase();
        String key = null;
        String rangeEnd = null;
        boolean prefix = false;
        int limit = 0;
        for (int index = 1; index < parts.size(); index++) {
            String part = parts.get(index);
            if (part.equals("--prefix")) {
                prefix = true;
            } else if (part.equals("--limit")) {
                if (index + 1 >= parts.size()) {
                    throw new IllegalArgumentException("--limit needs a value");
                }
                limit = Integer.parseInt(parts.get(++index));
            } else if (part.startsWith("--")) {
                throw new IllegalArgumentException("unsupported flag: " + part);
            } else if (key == null) {
                key = part;
            } else if (rangeEnd == null) {
                rangeEnd = part;
            } else {
                throw new IllegalArgumentException("too many arguments");
            }
        }
        return new EtcdCommand(operation, key, rangeEnd, prefix, limit);
    }
}
