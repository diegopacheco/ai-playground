package com.game.terminator.agent;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

public class AgentRunner {

    private static final Logger LOG = Logger.getLogger(AgentRunner.class.getName());

    private final String name;
    private final String model;

    public AgentRunner(String name, String model) {
        this.name = name;
        this.model = model;
    }

    public String run(String prompt) {
        List<String> command = buildCommand(prompt);
        LOG.info("[" + name + "/" + model + "] calling CLI...");
        long start = System.currentTimeMillis();
        try {
            ProcessBuilder pb = new ProcessBuilder(command);
            pb.redirectErrorStream(true);
            Process process = pb.start();
            StringBuilder output = new StringBuilder();
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    output.append(line).append("\n");
                }
            }
            boolean finished = process.waitFor(10, TimeUnit.SECONDS);
            long elapsed = System.currentTimeMillis() - start;
            if (!finished) {
                process.destroyForcibly();
                LOG.warning("[" + name + "/" + model + "] TIMEOUT after " + elapsed + "ms");
                return "";
            }
            String result = output.toString().trim();
            LOG.info("[" + name + "/" + model + "] response in " + elapsed + "ms: " + result);
            return result;
        } catch (Exception e) {
            long elapsed = System.currentTimeMillis() - start;
            LOG.severe("[" + name + "/" + model + "] ERROR after " + elapsed + "ms: " + e.getMessage());
            return "";
        }
    }

    private List<String> buildCommand(String prompt) {
        return switch (name) {
            case "claude" -> List.of("claude", "-p", prompt, "--model", model, "--dangerously-skip-permissions");
            case "gemini" -> List.of("gemini", "-y", "-p", prompt);
            case "copilot" -> List.of("copilot", "--allow-all", "--model", model, "-p", prompt);
            case "codex" -> List.of("codex", "exec", "--full-auto", "-m", model, prompt);
            default -> List.of("echo", "{}");
        };
    }

    public String getName() { return name; }
    public String getModel() { return model; }
}
