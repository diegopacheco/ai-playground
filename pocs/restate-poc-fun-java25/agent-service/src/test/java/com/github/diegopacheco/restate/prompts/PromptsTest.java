package com.github.diegopacheco.restate.prompts;

import com.github.diegopacheco.restate.AgentException;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PromptsTest {

    private static final String TEMPLATE = "---\nname: t\nparams: $goal, $goal_detail\n---\n\nGoal $goal with $goal_detail\n";

    @Test
    void replacesLongerNamesFirstSoAPrefixParamDoesNotCorruptIt() {
        String text = Prompts.render(TEMPLATE, Map.of("goal", "G", "goal_detail", "D"));
        assertEquals("Goal G with D", text);
    }

    @Test
    void failsLoudWhenADeclaredParamIsBlankSoTheAgentNeverGetsAHalfPrompt() {
        AgentException error = assertThrows(AgentException.class, () -> Prompts.render(TEMPLATE, Map.of("goal", "G", "goal_detail", " ")));
        assertEquals("prompt parameters without value: $goal_detail", error.getMessage());
    }

    @Test
    void rejectsATemplateWithoutHeaderBecauseItsParamsAreUndocumented() {
        assertThrows(AgentException.class, () -> Prompts.render("Goal $goal", Map.of("goal", "G")));
    }

    @Test
    void everyShippedPromptRendersWithTheParamsTheWorkflowSends() {
        Prompts prompts = new Prompts(Path.of("../prompts"));
        assertTrue(prompts.load("plan-goal", Map.of("goal", "octopus facts")).contains("octopus facts"));
        String task = prompts.load("execute-task", Map.of("goal", "g", "task", "find one fact", "previous_work", "none yet"));
        assertTrue(task.contains("find one fact") && task.contains("none yet"));
        assertTrue(prompts.load("summarize-run", Map.of("goal", "g", "work", "Task 1: x")).contains("Task 1: x"));
    }
}
