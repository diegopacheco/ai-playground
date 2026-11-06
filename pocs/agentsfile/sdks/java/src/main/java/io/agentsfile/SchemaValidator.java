package io.agentsfile;

import io.agentsfile.Statement.Attribute;
import io.agentsfile.Statement.Block;
import io.agentsfile.Value.ListValue;
import io.agentsfile.Value.MapValue;
import io.agentsfile.Value.StringValue;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

public final class SchemaValidator {
    private enum Kind {
        STRING("a string"),
        LIST("a list"),
        MAP("a map");

        private final String article;

        Kind(String article) {
            this.article = article;
        }

        boolean matches(Value value) {
            return switch (value) {
                case StringValue _ -> this == STRING;
                case ListValue _ -> this == LIST;
                case MapValue _ -> this == MAP;
            };
        }
    }

    private static final Map<String, Kind> FILE = Map.of("version", Kind.STRING);
    private static final Map<String, Kind> PROFILE = Map.of("description", Kind.STRING);
    private static final Map<String, Kind> SKILL = Map.of("source", Kind.STRING, "description", Kind.STRING);
    private static final Map<String, Kind> COMMAND = SKILL;
    private static final Map<String, Kind> AGENT = Map.of(
            "source", Kind.STRING, "description", Kind.STRING, "model", Kind.STRING, "tools", Kind.LIST);
    private static final Map<String, Kind> MCP = Map.of(
            "transport", Kind.STRING, "description", Kind.STRING, "command", Kind.STRING, "args", Kind.LIST,
            "env", Kind.MAP, "url", Kind.STRING, "headers", Kind.MAP);

    public Agentsfile validate(List<Statement> statements) {
        Attributes attributes = new Attributes("file", FILE);
        List<Profile> profiles = new ArrayList<>();
        Set<String> names = new HashSet<>();
        for (Statement statement : statements) {
            switch (statement) {
                case Attribute attribute -> attributes.add(attribute);
                case Block block -> {
                    if (!block.name().equals("profile")) {
                        throw unknownBlock(block, "file");
                    }
                    if (!names.add(block.label())) {
                        throw error(block, "duplicate profile \"" + block.label() + "\"");
                    }
                    profiles.add(profile(block));
                }
            }
        }
        String version = attributes.string("version")
                .orElseThrow(() -> new AgentsfileException(1, 1, "missing required attribute \"version\""));
        return new Agentsfile(version, profiles);
    }

    private Profile profile(Block profile) {
        Attributes attributes = new Attributes("profile", PROFILE);
        List<Skill> skills = new ArrayList<>();
        List<Agent> agents = new ArrayList<>();
        List<Command> commands = new ArrayList<>();
        List<Mcp> mcps = new ArrayList<>();
        Map<String, Set<String>> names = new LinkedHashMap<>();
        for (Statement statement : profile.body()) {
            switch (statement) {
                case Attribute attribute -> attributes.add(attribute);
                case Block block -> {
                    if (!List.of("skill", "agent", "command", "mcp").contains(block.name())) {
                        throw unknownBlock(block, "profile");
                    }
                    if (!names.computeIfAbsent(block.name(), k -> new HashSet<>()).add(block.label())) {
                        throw error(block, "duplicate " + block.name() + " \"" + block.label()
                                + "\" in profile \"" + profile.label() + "\"");
                    }
                    switch (block.name()) {
                        case "skill" -> skills.add(skill(block));
                        case "agent" -> agents.add(agent(block));
                        case "command" -> commands.add(command(block));
                        default -> mcps.add(mcp(block));
                    }
                }
            }
        }
        return new Profile(profile.label(), attributes.string("description"), skills, agents, commands, mcps);
    }

    private Skill skill(Block block) {
        Attributes attributes = leaf(block, SKILL);
        return new Skill(block.label(), required(attributes, block, "source"), attributes.string("description"));
    }

    private Command command(Block block) {
        Attributes attributes = leaf(block, COMMAND);
        return new Command(block.label(), required(attributes, block, "source"), attributes.string("description"));
    }

    private Agent agent(Block block) {
        Attributes attributes = leaf(block, AGENT);
        return new Agent(block.label(), required(attributes, block, "source"), attributes.string("description"),
                attributes.string("model"), attributes.list("tools"));
    }

    private Mcp mcp(Block block) {
        Attributes attributes = leaf(block, MCP);
        Transport transport = Transport.fromKeyword(required(attributes, block, "transport")).orElseThrow();
        String needed = transport == Transport.STDIO ? "command" : "url";
        if (attributes.string(needed).isEmpty()) {
            throw error(block, "mcp \"" + block.label() + "\" with " + transport.keyword()
                    + " transport requires \"" + needed + "\"");
        }
        return new Mcp(block.label(), transport, attributes.string("description"), attributes.string("command"),
                attributes.list("args"), attributes.map("env"), attributes.string("url"), attributes.map("headers"));
    }

    private Attributes leaf(Block block, Map<String, Kind> allowed) {
        Attributes attributes = new Attributes(block.name(), allowed);
        for (Statement statement : block.body()) {
            switch (statement) {
                case Attribute attribute -> attributes.add(attribute);
                case Block child -> throw unknownBlock(child, block.name());
            }
        }
        return attributes;
    }

    private static String required(Attributes attributes, Block block, String name) {
        return attributes.string(name).orElseThrow(() -> error(block,
                "missing required attribute \"" + name + "\" in " + block.name() + " \"" + block.label() + "\""));
    }

    private static AgentsfileException unknownBlock(Block block, String context) {
        return error(block, "unknown block \"" + block.name() + "\" in " + context);
    }

    private static AgentsfileException error(Statement statement, String message) {
        return new AgentsfileException(statement.line(), statement.col(), message);
    }

    private static final class Attributes {
        private final String context;
        private final Map<String, Kind> allowed;
        private final Map<String, Value> values = new LinkedHashMap<>();

        Attributes(String context, Map<String, Kind> allowed) {
            this.context = context;
            this.allowed = allowed;
        }

        void add(Attribute attribute) {
            String name = attribute.name();
            Kind kind = allowed.get(name);
            if (kind == null) {
                throw error(attribute, "unknown attribute \"" + name + "\" in " + context);
            }
            if (values.containsKey(name)) {
                throw error(attribute, "duplicate attribute \"" + name + "\" in " + context);
            }
            if (!kind.matches(attribute.value())) {
                throw error(attribute, "attribute \"" + name + "\" must be " + kind.article);
            }
            checkValue(attribute);
            values.put(name, attribute.value());
        }

        private void checkValue(Attribute attribute) {
            if (!(attribute.value() instanceof StringValue value)) {
                return;
            }
            if (context.equals("file") && attribute.name().equals("version") && !value.value().equals("1")) {
                throw new AgentsfileException(value.line(), value.col(),
                        "unsupported version \"" + value.value() + "\"");
            }
            if (context.equals("mcp") && attribute.name().equals("transport")
                    && Transport.fromKeyword(value.value()).isEmpty()) {
                throw new AgentsfileException(value.line(), value.col(),
                        "invalid transport \"" + value.value() + "\", expected stdio, http or sse");
            }
        }

        Optional<String> string(String name) {
            return Optional.ofNullable(values.get(name)).map(v -> ((StringValue) v).value());
        }

        List<String> list(String name) {
            Value value = values.get(name);
            return value == null ? List.of() : ((ListValue) value).items().stream().map(StringValue::value).toList();
        }

        Map<String, String> map(String name) {
            Map<String, String> result = new LinkedHashMap<>();
            if (values.get(name) instanceof MapValue value) {
                value.entries().forEach((key, entry) -> result.put(key, entry.value()));
            }
            return result;
        }
    }
}
