package com.github.diegopacheco.adminconsole.ai;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

@Component
public class AgentCliRunner {
    static final int MAX_PROMPT_BYTES = 16 * 1024;
    private static final int MAX_OUTPUT_BYTES = 256 * 1024;

    private final CliAvailability availability;
    private final long timeoutSeconds;

    public AgentCliRunner(CliAvailability availability,
                          @Value("${app.ai.timeout-seconds:60}") long timeoutSeconds) {
        this.availability = availability;
        this.timeoutSeconds = timeoutSeconds;
    }

    public List<String> command(AgentCli cli, String model, String prompt) {
        List<String> command = new ArrayList<>();
        command.add(cli.binary());
        command.addAll(cli.leadingArguments());
        if (model != null && !model.isBlank()) {
            command.add("--model");
            command.add(model);
        }
        command.add(prompt);
        return List.copyOf(command);
    }

    public String run(AgentCli cli, String model, String prompt) {
        if (prompt == null || prompt.isBlank()) {
            throw new IllegalArgumentException("prompt is required");
        }
        if (prompt.getBytes(StandardCharsets.UTF_8).length > MAX_PROMPT_BYTES) {
            throw new IllegalArgumentException("prompt is larger than " + MAX_PROMPT_BYTES + " bytes");
        }
        CliAvailability.Availability found = availability.of(cli);
        if (!found.available()) {
            throw new IllegalStateException(found.reason());
        }
        Process process = null;
        try {
            ProcessBuilder builder = new ProcessBuilder(command(cli, model, prompt));
            builder.redirectErrorStream(true);
            builder.redirectInput(ProcessBuilder.Redirect.from(new java.io.File("/dev/null")));
            process = builder.start();
            String output = read(process.getInputStream());
            if (!process.waitFor(timeoutSeconds, TimeUnit.SECONDS)) {
                process.destroyForcibly();
                throw new IllegalStateException(cli.binary() + " did not answer within " + timeoutSeconds + "s");
            }
            if (process.exitValue() != 0) {
                throw new IllegalStateException(cli.binary() + " failed: " + summarize(output));
            }
            return output;
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException(cli.binary() + " was interrupted", error);
        } catch (IOException error) {
            throw new IllegalStateException(cli.binary() + " could not be started: " + error.getMessage(), error);
        } finally {
            if (process != null && process.isAlive()) {
                process.destroyForcibly();
            }
        }
    }

    private String read(InputStream stream) throws IOException {
        byte[] bytes = stream.readNBytes(MAX_OUTPUT_BYTES);
        return new String(bytes, StandardCharsets.UTF_8);
    }

    private String summarize(String output) {
        String trimmed = output == null ? "" : output.trim();
        return trimmed.length() > 300 ? trimmed.substring(0, 300) + "…" : trimmed;
    }
}
