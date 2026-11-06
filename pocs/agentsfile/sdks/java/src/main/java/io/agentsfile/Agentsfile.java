package io.agentsfile;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

public record Agentsfile(String version, List<Profile> profiles) {
    public Agentsfile {
        profiles = List.copyOf(profiles);
    }

    public static Agentsfile parse(String source) {
        List<Token> tokens = new Lexer(source).tokenize();
        List<Statement> statements = new Parser(tokens).parse();
        return new SchemaValidator().validate(statements);
    }

    public static Agentsfile load(Path path) {
        try {
            return parse(Files.readString(path, StandardCharsets.UTF_8));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }
}
