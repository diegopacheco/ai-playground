package io.factoryfile;

import java.util.List;
import java.util.stream.Collectors;

final class Heredoc {
    private Heredoc() {
    }

    static String dedent(List<String> lines) {
        int indent = lines.stream()
                .filter(line -> !isBlank(line))
                .mapToInt(Heredoc::leadingIndent)
                .min()
                .orElse(0);
        return lines.stream()
                .map(line -> isBlank(line) ? "" : stripTrailingReturns(line.substring(indent)))
                .collect(Collectors.joining("\n"));
    }

    static boolean isBlankChar(char c) {
        return c == ' ' || c == '\t' || c == '\r';
    }

    private static boolean isBlank(String line) {
        return line.chars().allMatch(c -> isBlankChar((char) c));
    }

    private static int leadingIndent(String line) {
        int count = 0;
        while (count < line.length() && (line.charAt(count) == ' ' || line.charAt(count) == '\t')) {
            count++;
        }
        return count;
    }

    private static String stripTrailingReturns(String line) {
        int end = line.length();
        while (end > 0 && line.charAt(end - 1) == '\r') {
            end--;
        }
        return line.substring(0, end);
    }
}
