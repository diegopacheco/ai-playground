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

    private record Reference(String target, List<String> names, String owner, int line, int col) {
    }

    private static final Map<String, Kind> FILE = Map.of("version", Kind.STRING);
    private static final Map<String, Kind> FACTORY = Map.of(
            "title", Kind.STRING, "description", Kind.STRING, "entry", Kind.STRING, "flow", Kind.STRING,
            "backlog", Kind.STRING);
    private static final Map<String, Kind> AGENT = Map.of(
            "description", Kind.STRING, "role", Kind.STRING, "provider", Kind.STRING, "model", Kind.STRING,
            "tools", Kind.LIST, "source", Kind.STRING, "content", Kind.STRING);
    private static final Map<String, Kind> STAGE = Map.ofEntries(
            Map.entry("label", Kind.STRING), Map.entry("description", Kind.STRING),
            Map.entry("agents", Kind.LIST), Map.entry("run", Kind.STRING), Map.entry("script", Kind.STRING),
            Map.entry("next", Kind.LIST), Map.entry("artifacts", Kind.LIST), Map.entry("output", Kind.STRING),
            Map.entry("check", Kind.STRING), Map.entry("verdict", Kind.MAP), Map.entry("source", Kind.STRING),
            Map.entry("content", Kind.STRING), Map.entry("skills", Kind.LIST), Map.entry("tasks", Kind.STRING),
            Map.entry("parallel", Kind.STRING), Map.entry("visits", Kind.STRING), Map.entry("end", Kind.STRING),
            Map.entry("at", Kind.LIST));
    private static final Map<String, Kind> ON = Map.of(
            "to", Kind.STRING, "label", Kind.STRING, "reason", Kind.STRING, "cycle", Kind.STRING);
    private static final Map<String, Kind> EXHAUSTED = Map.of("label", Kind.STRING, "reason", Kind.STRING);
    private static final Map<String, Kind> LOOP = Map.of(
            "description", Kind.STRING, "from", Kind.STRING, "to", Kind.STRING, "until", Kind.STRING,
            "max", Kind.STRING);
    private static final List<String> FACTORY_KINDS = List.of("agent", "stage", "loop");
    private static final Map<String, String> REFERENCES = Map.of(
            "factory.entry", "stage",
            "stage.agents", "agent",
            "stage.next", "stage",
            "stage.tasks", "stage",
            "on.to", "stage",
            "loop.from", "stage",
            "loop.to", "stage");
    private static final Pattern MAX = Pattern.compile("[1-9][0-9]{0,5}");
    private static final Pattern AT = Pattern.compile("0|[1-9][0-9]{0,2}");
    private static final Set<String> VERDICT_KEYS = Set.of("field", "yes", "no", "list", "notes");

    private final List<Reference> references = new ArrayList<>();

    public Factoryfile validate(List<Statement> statements) {
        List<Factory> factories = new ArrayList<>();
        Set<String> names = new HashSet<>();
        Attributes attributes = walk("file", "file \"\"", statements, FILE, block -> {
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
        Attributes attributes = walk("factory", owner(block), block.body(), FACTORY, child -> {
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
        return new Factory(block.label(), attributes.string("title"), attributes.string("description"), entry,
                attributes.string("flow"), attributes.string("backlog"), agents, stages, loops);
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
                attributes.string("provider"), attributes.string("model"), attributes.list("tools"), attributes.string("source"),
                attributes.string("content"));
    }

    private Stage stage(Block block) {
        String owner = owner(block);
        List<Route> routes = new ArrayList<>();
        Set<String> outcomes = new HashSet<>();
        Attributes attributes = walk("stage", owner, block.body(), STAGE, child -> {
            if (!child.name().equals("on")) {
                throw unknownBlock(child, "stage");
            }
            if (!outcomes.add(child.label())) {
                throw error(child, "duplicate on \"" + child.label() + "\" in stage \"" + block.label() + "\"");
            }
            routes.add(route(child, owner));
        });
        if (!attributes.has("agents") && !attributes.has("run")) {
            throw missing(block, "\"agents\" or \"run\"");
        }
        if (attributes.has("source") && attributes.has("content")) {
            throw error(block, "stage \"" + block.label() + "\" cannot set both \"source\" and \"content\"");
        }
        if (attributes.has("next") && outcomes.contains("next")) {
            throw error(block, "stage \"" + block.label() + "\" cannot set both \"next\" and on \"next\"");
        }
        if (attributes.has("next") || outcomes.contains("next")) {
            routes.stream().filter(r -> !r.outcome().equals("next")).findFirst().ifPresent(r -> {
                throw error(block, "stage \"" + block.label() + "\" cannot mix the next route with on \""
                        + r.outcome() + "\"");
            });
        }
        return new Stage(block.label(), attributes.string("label"), attributes.string("description"),
                attributes.list("agents"), attributes.string("run"), attributes.string("script"),
                attributes.list("next"), attributes.list("artifacts"), attributes.string("output"),
                attributes.string("check"), attributes.map("verdict"), attributes.string("source"),
                attributes.string("content"), attributes.list("skills"), attributes.string("tasks"),
                attributes.number("parallel"), attributes.number("visits"), attributes.string("end"),
                attributes.list("at").stream().map(Integer::parseInt).toList(), routes);
    }

    private Route route(Block block, String owner) {
        List<Exhausted> fallback = new ArrayList<>();
        Attributes attributes = walk("on", owner, block.body(), ON, exhaustedBlock(block, owner, fallback));
        if (!attributes.has("to")) {
            throw missing(block, "\"to\"");
        }
        return new Route(block.label(), attributes.string("to").orElseThrow(), attributes.string("label"),
                attributes.string("reason"), attributes.string("cycle").filter("true"::equals).isPresent(),
                fallback.stream().findFirst());
    }

    private Consumer<Block> exhaustedBlock(Block parent, String owner, List<Exhausted> into) {
        return child -> {
            if (!child.name().equals("exhausted")) {
                throw unknownBlock(child, parent.name());
            }
            if (!into.isEmpty()) {
                throw error(child, "duplicate exhausted in " + parent.name() + " \"" + parent.label() + "\"");
            }
            references.add(new Reference("stage", List.of(child.label()), owner, child.line(), child.col()));
            into.add(exhausted(child, owner));
        };
    }

    private Exhausted exhausted(Block block, String owner) {
        List<Exhausted> next = new ArrayList<>();
        Attributes attributes = walk("exhausted", owner, block.body(), EXHAUSTED, exhaustedBlock(block, owner, next));
        return new Exhausted(block.label(), attributes.string("label"), attributes.string("reason"),
                next.stream().findFirst());
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
                attributes.number("max"));
    }

    private Attributes leaf(Block block, Map<String, Kind> allowed) {
        return walk(block.name(), owner(block), block.body(), allowed, child -> {
            throw unknownBlock(child, block.name());
        });
    }

    private static String owner(Block block) {
        return block.name() + " \"" + block.label() + "\"";
    }

    private Attributes walk(String context, String owner, List<Statement> body, Map<String, Kind> allowed,
                            Consumer<Block> onBlock) {
        Attributes attributes = new Attributes(context, allowed);
        for (Statement statement : body) {
            switch (statement) {
                case Attribute attribute -> {
                    attributes.add(attribute);
                    collectReference(context, owner, attribute);
                }
                case Block block -> onBlock.accept(block);
            }
        }
        return attributes;
    }

    private void collectReference(String context, String owner, Attribute attribute) {
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
        references.add(new Reference(target, names, owner, value.line(), value.col()));
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
            Value value = attribute.value();
            switch (context + "." + attribute.name()) {
                case "file.version" -> {
                    String version = ((StringValue) value).value();
                    if (!version.equals("1")) {
                        throw invalid(value, "unsupported version \"" + version + "\"");
                    }
                }
                case "loop.max", "stage.parallel", "stage.visits" -> {
                    String number = ((StringValue) value).value();
                    if (!MAX.matcher(number).matches()) {
                        throw invalid(value, "invalid " + attribute.name() + " \"" + number
                                + "\", expected an integer from 1 to 999999");
                    }
                }
                case "stage.at" -> {
                    List<StringValue> items = ((ListValue) value).items();
                    if (items.size() != 2 || !items.stream().allMatch(i -> AT.matcher(i.value()).matches())) {
                        throw invalid(value, "invalid at, expected two integers from 0 to 999");
                    }
                }
                case "on.cycle" -> {
                    String cycle = ((StringValue) value).value();
                    if (!cycle.equals("true") && !cycle.equals("false")) {
                        throw invalid(value, "invalid cycle \"" + cycle + "\", expected \"true\" or \"false\"");
                    }
                }
                case "stage.verdict" -> {
                    Map<String, StringValue> entries = ((MapValue) value).entries();
                    entries.keySet().stream().sorted().filter(k -> !VERDICT_KEYS.contains(k)).findFirst()
                            .ifPresent(k -> {
                                throw invalid(value, "unknown key \"" + k + "\" in verdict");
                            });
                    if (!entries.containsKey("field")) {
                        throw invalid(value, "verdict requires \"field\"");
                    }
                }
                default -> {
                }
            }
        }

        private static FactoryfileException invalid(Value value, String message) {
            return new FactoryfileException(value.line(), value.col(), message);
        }

        boolean has(String name) {
            return values.containsKey(name);
        }

        Optional<String> string(String name) {
            return Optional.ofNullable(values.get(name)).map(v -> ((StringValue) v).value());
        }

        Optional<Integer> number(String name) {
            return string(name).map(Integer::parseInt);
        }

        Optional<Map<String, String>> map(String name) {
            return Optional.ofNullable((MapValue) values.get(name)).map(v -> {
                Map<String, String> out = new LinkedHashMap<>();
                v.entries().forEach((key, entry) -> out.put(key, entry.value()));
                return out;
            });
        }

        List<String> list(String name) {
            Value value = values.get(name);
            return value == null ? List.of() : ((ListValue) value).items().stream().map(StringValue::value).toList();
        }
    }
}
