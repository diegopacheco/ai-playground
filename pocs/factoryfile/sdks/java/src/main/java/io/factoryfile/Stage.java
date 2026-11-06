package io.factoryfile;

import java.util.List;
import java.util.Map;
import java.util.Optional;

public record Stage(
        String name,
        Optional<String> label,
        Optional<String> description,
        List<String> agents,
        Optional<String> run,
        Optional<String> script,
        List<String> next,
        List<String> artifacts,
        Optional<String> output,
        Optional<String> check,
        Optional<Map<String, String>> verdict,
        Optional<String> source,
        Optional<String> content,
        List<String> skills,
        Optional<String> tasks,
        Optional<Integer> parallel,
        Optional<Integer> visits,
        Optional<String> end,
        List<Integer> at,
        List<Route> routes) {
    public Stage {
        agents = List.copyOf(agents);
        next = List.copyOf(next);
        artifacts = List.copyOf(artifacts);
        verdict = verdict.map(Map::copyOf);
        skills = List.copyOf(skills);
        at = List.copyOf(at);
        routes = List.copyOf(routes);
    }
}
