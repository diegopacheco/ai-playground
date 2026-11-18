package com.github.diegopacheco.restate.workflow;

import com.github.diegopacheco.restate.agents.AgentCaller;
import com.github.diegopacheco.restate.config.AgentConfig;
import com.github.diegopacheco.restate.plan.Plan;
import com.github.diegopacheco.restate.plan.PlanParser;
import com.github.diegopacheco.restate.prompts.Prompts;
import com.github.diegopacheco.restate.state.RunState;
import dev.restate.sdk.SharedWorkflowContext;
import dev.restate.sdk.WorkflowContext;
import dev.restate.sdk.annotation.Shared;
import dev.restate.sdk.annotation.Workflow;
import dev.restate.sdk.common.RetryPolicy;
import dev.restate.sdk.common.StateKey;
import dev.restate.sdk.common.TerminalException;
import org.springframework.stereotype.Component;

import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Optional;

@Component
@Workflow
public class AgentWorkflow {

    private static final StateKey<RunState> STATE = StateKey.of("state", RunState.class);
    private static final RetryPolicy RETRY = RetryPolicy.exponential(Duration.ofSeconds(2), 2.0f).setMaxAttempts(3);

    private final AgentCaller agent;
    private final Prompts prompts;

    public AgentWorkflow(AgentCaller agent, Prompts prompts) {
        this.agent = agent;
        this.prompts = prompts;
    }

    @Workflow
    public String run(WorkflowContext ctx, RunRequest request) {
        String goal = Optional.ofNullable(request.goal()).orElse("").trim();
        if (goal.isEmpty()) {
            throw new TerminalException(400, "goal is required");
        }
        AgentConfig config = ctx.run("read config", AgentConfig.class, () -> AgentConfig.from(System::getenv));
        RunState state = RunState.start(goal, config.provider().cli(), config.model());
        ctx.set(STATE, state);
        try {
            return runSteps(ctx, config, goal, state);
        } catch (TerminalException error) {
            RunState latest = ctx.get(STATE).orElse(state);
            ctx.set(STATE, latest.failed(error.getMessage()));
            throw error;
        }
    }

    @Shared
    public RunState status(SharedWorkflowContext ctx) {
        return ctx.get(STATE).orElse(null);
    }

    private String runSteps(WorkflowContext ctx, AgentConfig config, String goal, RunState initial) {
        Plan plan = ctx.run("plan", Plan.class, RETRY,
                () -> PlanParser.parse(agent.ask(config, prompts.load("plan-goal", Map.of("goal", goal)))));
        RunState state = save(ctx, initial.withPlan(plan.tasks()));
        List<String> tasks = plan.tasks();
        for (int index = 0; index < tasks.size(); index++) {
            state = save(ctx, state.stepRunning(index));
            String previous = state.completedWork().isEmpty() ? "none yet" : state.completedWork();
            String prompt = prompts.load("execute-task", Map.of("goal", goal, "task", tasks.get(index), "previous_work", previous));
            String output = ctx.run("task " + (index + 1), String.class, RETRY, () -> agent.ask(config, prompt));
            state = save(ctx, state.stepDone(index, output));
        }
        state = save(ctx, state.summarizing());
        String work = state.completedWork();
        String summary = ctx.run("summarize", String.class, RETRY,
                () -> agent.ask(config, prompts.load("summarize-run", Map.of("goal", goal, "work", work))));
        save(ctx, state.finished(summary));
        return summary;
    }

    private static RunState save(WorkflowContext ctx, RunState state) {
        ctx.set(STATE, state);
        return state;
    }
}
