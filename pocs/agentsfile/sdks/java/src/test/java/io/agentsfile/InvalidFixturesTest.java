package io.agentsfile;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class InvalidFixturesTest {
    private static final String PREFIX = "# expect: ";

    static List<Path> fixtures() {
        return Fixtures.invalid();
    }

    @Test
    void everyInvalidFixtureIsDiscoveredSoNoneIsSilentlySkipped() {
        assertEquals(51, fixtures().size());
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("fixtures")
    void rejectsWithTheExactSpecMessageAndPosition(Path fixture) throws IOException {
        String firstLine = Files.readAllLines(fixture).getFirst();
        assertTrue(firstLine.startsWith(PREFIX), "fixture lacks expectation header: " + fixture);
        String expected = firstLine.substring(PREFIX.length());

        AgentsfileException error = assertThrows(AgentsfileException.class, () -> Agentsfile.load(fixture));

        assertEquals(expected, error.getMessage());
        assertEquals(expected, "line " + error.line() + ", col " + error.col() + ": "
                + expected.substring(expected.indexOf(": ") + 2));
    }
}
