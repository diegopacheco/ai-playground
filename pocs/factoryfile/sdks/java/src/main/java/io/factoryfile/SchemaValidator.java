package io.factoryfile;

import io.factoryfile.Statement.Attribute;
import io.factoryfile.Statement.Block;
import io.factoryfile.Value.ListValue;
import io.factoryfile.Value.MapValue;
import io.factoryfile.Value.StringValue;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.function.Consumer;
import java.util.regex.Pattern;

public final class SchemaValidator {
    private enum Kind {
        STRING("a string"),
        LIST("a list");

        private final String article;

        Kind(String article) {
            this.article = article;
        }

        boolean matches(Value value) {
            return switch (value) {
                case StringValue _ -> this == STRING;
                case ListValue _ -> this == LIST;
                case MapValue _ -> false;
            };
        }
    }

    private record Reference(String target, List<String> names, String owner, int line, int col) {
    }

    private static final Map<String, Kind> FILE = Map.of("version", Kind.STRING);
    private static final Map<String, Kind> FACTORY = Map.of("description", Kind.STRING, "entry", Kind.STRING);
    private static final Map<String, Kind> AGENT = Map.of(
            "description", Kind.STRING, "role", Kind.STRING, "model", Kind.STRING, "tools", Kind.LIST,
            "source", Kind.STRING, "content", Kind.STRING);
    private static final Map<String, Kind> STAGE = Map.of(
            "description", Kind.STRING, "agents", Kind.LIST, "next", Kind.LIST, "artifacts", Kind.LIST,
            "content", Kind.STRING);
    private static final Map<String, Kind> LOOP = Map.of(
            "description", Kind.STRING, "from", Kind.STRING, "to", Kind.STRING, "until", Kind.STRING,
            "max", Kind.STRING);
    private static final List<String> FACTORY_KINDS = List.of("agent", "stage", "loop");
    private static final Map<String, String> REFERENCES = Map.of(
            "factory.entry", "stage",
            "stage.agents", "agent",
            "stage.next", "stage",
            "loop.from", "stage",
            "loop.to", "stage");
    private static final Pattern MAX = Pattern.compile("[1-9][0-9]{0,5}");

    private final List<Reference> references = new ArrayList<>();

    public Factoryfile validate(List<Statement> statements) {
        List<Factory> factories = new ArrayList<>();
        Set<String> names = new HashSet<>();
        Attributes attributes = walk("file", "", statements, FILE, block -> {
            if (!block.name().equals("factory")) {
                throw unknownBlock(block, "file");
            }
            if (!names.add(block.label())) {
                throw error(block, "duplicate factory \"" + block.label() + "\"");
            }
            factories.add(factory(block));
        });
        String version = attributes.string("version")
                .orElseThrow(() -> new FactoryfileException(1, 1, "missing required attribute \"version\""));
        return new Factoryfile(version, factories);
    }

    private Factory factory(Block block) {
        references.clear();
        List<Agent> agents = new ArrayList<>();
        List<Stage> stages = new ArrayList<>();
        List<Loop> loops = new ArrayList<>();
        Map<String, Set<String>> defined = new HashMap<>();
        Attributes attributes = walk("factory", block.label(), block.body(), FACTORY, child -> {
            if (!FACTORY_KINDS.contains(child.name())) {
                throw unknownBlock(child, "factory");
            }
            if (!defined.computeIfAbsent(child.name(), k -> new HashSet<>()).add(child.label())) {
                throw error(child, "duplicate " + child.name() + " \"" + child.label() + "\" in factory \""
                        + block.label() + "\"");
            }
            switch (child.name()) {
                case "agent" -> agents.add(agent(child));
                case "stage" -> stages.add(stage(child));
                default -> loops.add(loop(child));
            }
        });
        checkReferences(defined);
        Optional<String> entry = attributes.string("entry").or(() -> stages.stream().findFirst().map(Stage::name));
        return new Factory(block.label(), attributes.string("description"), entry, agents, stages, loops);
    }

    private void checkReferences(Map<String, Set<String>> defined) {
        for (Reference reference : references) {
            Set<String> known = defined.getOrDefault(reference.target(), Set.of());
            for (String name : reference.names()) {
                if (!known.contains(name)) {
                    throw new FactoryfileException(reference.line(), reference.col(),
                            "unknown " + reference.target() + " \"" + name + "\" in " + reference.owner());
                }
            }
        }
    }

    private Agent agent(Block block) {
        Attributes attributes = leaf(block, AGENT);
        boolean hasSource = attributes.has("source");
        boolean hasContent = attributes.has("content");
        if (!hasSource && !hasContent) {
            throw missing(block, "\"source\" or \"content\"");
        }
        if (hasSource && hasContent) {
            throw error(block, "agent \"" + block.label() + "\" cannot set both \"source\" and \"content\"");
        }
        return new Agent(block.label(), attributes.string("description"), attributes.string("role"),
                attributes.string("model"), attributes.list("tools"), attributes.string("source"),
                attributes.string("content"));
    }

    private Stage stage(Block block) {
        Attributes attributes = leaf(block, STAGE);
        if (!attributes.has("agents")) {
            throw missing(block, "\"agents\"");
        }
        return new Stage(block.label(), attributes.string("description"), attributes.list("agents"),
                attributes.list("next"), attributes.list("artifacts"), attributes.string("content"));
    }

    private Loop loop(Block block) {
        Attributes attributes = leaf(block, LOOP);
        for (String required : List.of("from", "to")) {
            if (!attributes.has(required)) {
                throw missing(block, "\"" + required + "\"");
            }
        }
        return new Loop(block.label(), attributes.string("description"), attributes.string("from").orElseThrow(),
                attributes.string("to").orElseThrow(), attributes.string("until"),
                attributes.string("max").map(Integer::parseInt));
    }

    private Attributes leaf(Block block, Map<String, Kind> allowed) {
        return walk(block.name(), block.label(), block.body(), allowed, child -> {
            throw unknownBlock(child, block.name());
        });
    }

    private Attributes walk(String context, String label, List<Statement> body, Map<String, Kind> allowed,
                            Consumer<Block> onBlock) {
        Attributes attributes = new Attributes(context, allowed);
        for (Statement statement : body) {
            switch (statement) {
                case Attribute attribute -> {
                    attributes.add(attribute);
                    collectReference(context, label, attribute);
                }
                case Block block -> onBlock.accept(block);
            }
        }
        return attributes;
    }

    private void collectReference(String context, String label, Attribute attribute) {
        String target = REFERENCES.get(context + "." + attribute.name());
        if (target == null) {
            return;
        }
        Value value = attribute.value();
        List<String> names = switch (value) {
            case StringValue string -> List.of(string.value());
            case ListValue list -> list.items().stream().map(StringValue::value).toList();
            case MapValue _ -> List.of();
        };
        references.add(new Reference(target, names, context + " \"" + label + "\"", value.line(), value.col()));
    }

    private static FactoryfileException missing(Block block, String what) {
        return error(block, block.name() + " \"" + block.label() + "\" requires " + what);
    }

    private static FactoryfileException unknownBlock(Block block, String context) {
        return error(block, "unknown block \"" + block.name() + "\" in " + context);
    }

    private static FactoryfileException error(Statement statement, String message) {
        return new FactoryfileException(statement.line(), statement.col(), message);
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
                throw new FactoryfileException(value.line(), value.col(),
                        "unsupported version \"" + value.value() + "\"");
            }
            if (context.equals("loop") && attribute.name().equals("max") && !MAX.matcher(value.value()).matches()) {
                throw new FactoryfileException(value.line(), value.col(),
                        "invalid max \"" + value.value() + "\", expected an integer from 1 to 999999");
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
    }
}
