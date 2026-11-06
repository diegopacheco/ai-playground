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
import java.util.function.Consumer;

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
    private static final Map<String, Kind> SKILL = Map.of(
            "description", Kind.STRING, "source", Kind.STRING, "content", Kind.STRING);
    private static final Map<String, Kind> SKILL_FILE = Map.of("source", Kind.STRING, "content", Kind.STRING);
    private static final Map<String, Kind> AGENT = Map.of(
            "description", Kind.STRING, "model", Kind.STRING, "tools", Kind.LIST,
            "source", Kind.STRING, "content", Kind.STRING);
    private static final Map<String, Kind> COMMAND = Map.of(
            "description", Kind.STRING, "argument-hint", Kind.STRING, "model", Kind.STRING, "tools", Kind.LIST,
            "source", Kind.STRING, "content", Kind.STRING);
    private static final Map<String, Kind> MCP = Map.of(
            "description", Kind.STRING, "source", Kind.STRING, "transport", Kind.STRING, "command", Kind.STRING,
            "args", Kind.LIST, "env", Kind.MAP, "url", Kind.STRING, "headers", Kind.MAP);
    private static final Map<String, Kind> PLUGIN = Map.of(
            "description", Kind.STRING, "version", Kind.STRING, "author", Kind.STRING, "source", Kind.STRING);
    private static final List<String> PROFILE_KINDS = List.of("skill", "agent", "command", "mcp", "plugin");
    private static final List<String> PLUGIN_KINDS = List.of("skill", "agent", "command", "mcp");

    public Agentsfile validate(List<Statement> statements) {
        List<Profile> profiles = new ArrayList<>();
        Set<String> names = new HashSet<>();
        Attributes attributes = walk("file", statements, FILE, block -> {
            if (!block.name().equals("profile")) {
                throw unknownBlock(block, "file");
            }
            if (!names.add(block.label())) {
                throw error(block, "duplicate profile \"" + block.label() + "\"");
            }
            profiles.add(profile(block));
        });
        String version = attributes.string("version")
                .orElseThrow(() -> new AgentsfileException(1, 1, "missing required attribute \"version\""));
        return new Agentsfile(version, profiles);
    }

    private Profile profile(Block block) {
        Items items = new Items();
        Attributes attributes = container(block, PROFILE, PROFILE_KINDS, items);
        return new Profile(block.label(), attributes.string("description"),
                items.skills, items.agents, items.commands, items.mcps, items.plugins);
    }

    private Plugin plugin(Block block) {
        Items items = new Items();
        Attributes attributes = container(block, PLUGIN, PLUGIN_KINDS, items);
        exactlyOne(block, attributes.has("source"), !items.isEmpty(), "inline items");
        return new Plugin(block.label(), attributes.string("description"), attributes.string("version"),
                attributes.string("author"), attributes.string("source"),
                items.skills, items.agents, items.commands, items.mcps);
    }

    private Attributes container(Block owner, Map<String, Kind> allowed, List<String> kinds, Items items) {
        Map<String, Set<String>> names = new LinkedHashMap<>();
        return walk(owner.name(), owner.body(), allowed, block -> {
            if (!kinds.contains(block.name())) {
                throw unknownBlock(block, owner.name());
            }
            if (!names.computeIfAbsent(block.name(), k -> new HashSet<>()).add(block.label())) {
                throw duplicate(block, owner);
            }
            switch (block.name()) {
                case "skill" -> items.skills.add(skill(block));
                case "agent" -> items.agents.add(agent(block));
                case "command" -> items.commands.add(command(block));
                case "mcp" -> items.mcps.add(mcp(block));
                default -> items.plugins.add(plugin(block));
            }
        });
    }

    private Skill skill(Block block) {
        List<SkillFile> files = new ArrayList<>();
        Set<String> paths = new HashSet<>();
        Attributes attributes = walk("skill", block.body(), SKILL, child -> {
            if (!child.name().equals("file")) {
                throw unknownBlock(child, "skill");
            }
            if (!paths.add(child.label())) {
                throw duplicate(child, block);
            }
            files.add(skillFile(child));
        });
        sourceOrContent(block, attributes);
        return new Skill(block.label(), attributes.string("description"), attributes.string("source"),
                attributes.string("content"), files);
    }

    private SkillFile skillFile(Block block) {
        Attributes attributes = leaf(block, SKILL_FILE);
        sourceOrContent(block, attributes);
        return new SkillFile(block.label(), attributes.string("source"), attributes.string("content"));
    }

    private Agent agent(Block block) {
        Attributes attributes = leaf(block, AGENT);
        sourceOrContent(block, attributes);
        return new Agent(block.label(), attributes.string("description"), attributes.string("model"),
                attributes.list("tools"), attributes.string("source"), attributes.string("content"));
    }

    private Command command(Block block) {
        Attributes attributes = leaf(block, COMMAND);
        sourceOrContent(block, attributes);
        return new Command(block.label(), attributes.string("description"), attributes.string("argument-hint"),
                attributes.string("model"), attributes.list("tools"), attributes.string("source"),
                attributes.string("content"));
    }

    private Mcp mcp(Block block) {
        Attributes attributes = leaf(block, MCP);
        exactlyOne(block, attributes.has("source"), attributes.has("transport"), "\"transport\"");
        Optional<Transport> transport = attributes.string("transport").flatMap(Transport::fromKeyword);
        transport.ifPresent(t -> {
            String needed = t == Transport.STDIO ? "command" : "url";
            if (!attributes.has(needed)) {
                throw error(block, "mcp \"" + block.label() + "\" with " + t.keyword()
                        + " transport requires \"" + needed + "\"");
            }
        });
        return new Mcp(block.label(), attributes.string("description"), attributes.string("source"), transport,
                attributes.string("command"), attributes.list("args"), attributes.map("env"),
                attributes.string("url"), attributes.map("headers"));
    }

    private static void sourceOrContent(Block block, Attributes attributes) {
        exactlyOne(block, attributes.has("source"), attributes.has("content"), "\"content\"");
    }

    private static void exactlyOne(Block block, boolean hasSource, boolean hasInline, String inline) {
        String subject = block.name() + " \"" + block.label() + "\" ";
        if (!hasSource && !hasInline) {
            throw error(block, subject + "requires \"source\" or " + inline);
        }
        if (hasSource && hasInline) {
            throw error(block, subject + "cannot set both \"source\" and " + inline);
        }
    }

    private static Attributes leaf(Block block, Map<String, Kind> allowed) {
        return walk(block.name(), block.body(), allowed, child -> {
            throw unknownBlock(child, block.name());
        });
    }

    private static Attributes walk(String context, List<Statement> body, Map<String, Kind> allowed,
                                   Consumer<Block> onBlock) {
        Attributes attributes = new Attributes(context, allowed);
        for (Statement statement : body) {
            switch (statement) {
                case Attribute attribute -> attributes.add(attribute);
                case Block block -> onBlock.accept(block);
            }
        }
        return attributes;
    }

    private static AgentsfileException duplicate(Block block, Block owner) {
        return error(block, "duplicate " + block.name() + " \"" + block.label() + "\" in "
                + owner.name() + " \"" + owner.label() + "\"");
    }

    private static AgentsfileException unknownBlock(Block block, String context) {
        return error(block, "unknown block \"" + block.name() + "\" in " + context);
    }

    private static AgentsfileException error(Statement statement, String message) {
        return new AgentsfileException(statement.line(), statement.col(), message);
    }

    private static final class Items {
        private final List<Skill> skills = new ArrayList<>();
        private final List<Agent> agents = new ArrayList<>();
        private final List<Command> commands = new ArrayList<>();
        private final List<Mcp> mcps = new ArrayList<>();
        private final List<Plugin> plugins = new ArrayList<>();

        boolean isEmpty() {
            return skills.isEmpty() && agents.isEmpty() && commands.isEmpty() && mcps.isEmpty() && plugins.isEmpty();
        }
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

        boolean has(String name) {
            return values.containsKey(name);
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
