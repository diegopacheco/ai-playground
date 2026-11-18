package com.github.diegopacheco.restate.prompts;

import com.github.diegopacheco.restate.AgentException;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@Component
public class Prompts {

    private static final Pattern PARAM = Pattern.compile("\\$([a-z_]+)");
    private static final String SEPARATOR = "---\n";

    private final Path dir;

    public Prompts() {
        this(Path.of(Optional.ofNullable(System.getenv("PROMPTS_DIR")).orElse("../prompts")));
    }

    public Prompts(Path dir) {
        this.dir = dir;
    }

    public String load(String name, Map<String, String> params) {
        try {
            return render(Files.readString(dir.resolve(name + ".md")), params);
        } catch (IOException error) {
            throw new AgentException("cannot read prompt " + name + ": " + error.getMessage());
        }
    }

    public static String render(String template, Map<String, String> params) {
        int end = template.indexOf("\n" + SEPARATOR, SEPARATOR.length() - 1);
        if (!template.startsWith(SEPARATOR) || end < 0) {
            throw new AgentException("prompt is missing its --- header");
        }
        List<String> declared = declared(template.substring(SEPARATOR.length(), end));
        List<String> missing = declared.stream()
                .filter(name -> params.getOrDefault(name, "").isBlank())
                .map(name -> "$" + name)
                .toList();
        if (!missing.isEmpty()) {
            throw new AgentException("prompt parameters without value: " + String.join(", ", missing));
        }
        String body = template.substring(end + 1 + SEPARATOR.length());
        for (String name : declared.stream().sorted(Comparator.comparingInt(String::length).reversed()).toList()) {
            body = body.replace("$" + name, params.get(name));
        }
        return body.trim();
    }

    private static List<String> declared(String header) {
        Matcher matcher = PARAM.matcher(header);
        return matcher.results().map(result -> result.group(1)).toList();
    }
}
