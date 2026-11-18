package com.github.diegopacheco.restate.state;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public record RunState(String goal, String provider, String model, Phase phase, List<Step> steps, String summary, String error) {

    public static RunState start(String goal, String provider, String model) {
        return new RunState(goal, provider, model, Phase.PLANNING, List.of(), "", "");
    }

    public RunState withPlan(List<String> tasks) {
        List<Step> planned = tasks.stream().map(task -> new Step(task, StepStatus.PENDING, "")).toList();
        return new RunState(goal, provider, model, Phase.WORKING, planned, summary, error);
    }

    public RunState stepRunning(int index) {
        return withStep(index, StepStatus.RUNNING, steps.get(index).output());
    }

    public RunState stepDone(int index, String output) {
        return withStep(index, StepStatus.DONE, output);
    }

    public RunState summarizing() {
        return new RunState(goal, provider, model, Phase.SUMMARIZING, steps, summary, error);
    }

    public RunState finished(String text) {
        return new RunState(goal, provider, model, Phase.DONE, steps, text, error);
    }

    public RunState failed(String message) {
        return new RunState(goal, provider, model, Phase.FAILED, steps, summary, message);
    }

    public String completedWork() {
        List<Step> done = steps.stream().filter(step -> step.status() == StepStatus.DONE).toList();
        return IntStream.range(0, done.size())
                .mapToObj(i -> "Task " + (i + 1) + ": " + done.get(i).task() + "\nResult: " + done.get(i).output())
                .collect(Collectors.joining("\n\n"));
    }

    private RunState withStep(int index, StepStatus status, String output) {
        List<Step> updated = IntStream.range(0, steps.size())
                .mapToObj(i -> i == index ? new Step(steps.get(i).task(), status, output) : steps.get(i))
                .toList();
        return new RunState(goal, provider, model, phase, updated, summary, error);
    }
}
