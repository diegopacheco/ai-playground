package com.github.diegopacheco.devadminconsole.engine.redis;

import java.util.ArrayList;
import java.util.List;
import org.springframework.stereotype.Component;

@Component
public class RedisCommandParser {
    public List<String> parse(String command) {
        if (command == null || command.isBlank()) {
            throw new IllegalArgumentException("command is required");
        }
        List<String> values = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        char quote = 0;
        boolean started = false;
        for (int index = 0; index < command.trim().length(); index++) {
            char character = command.trim().charAt(index);
            if (quote != 0) {
                if (character == '\\' && index + 1 < command.trim().length()) {
                    current.append(command.trim().charAt(++index));
                    continue;
                }
                if (character == quote) {
                    quote = 0;
                    continue;
                }
                current.append(character);
                continue;
            }
            if (character == '"' || character == '\'') {
                quote = character;
                started = true;
                continue;
            }
            if (Character.isWhitespace(character)) {
                if (started || !current.isEmpty()) {
                    values.add(current.toString());
                    current.setLength(0);
                    started = false;
                }
                continue;
            }
            current.append(character);
        }
        if (quote != 0) {
            throw new IllegalArgumentException("unbalanced quote in command");
        }
        if (started || !current.isEmpty()) {
            values.add(current.toString());
        }
        if (values.isEmpty()) {
            throw new IllegalArgumentException("command is required");
        }
        return values;
    }
}
