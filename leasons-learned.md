# Lessons Learned After 500+ AI POCs

Over two years, from summer 2024 to summer 2026, I built more than 500 AI POCs and more than 100 agent POCs. The work covered local models, RAG, embeddings, MCP, coding agents, Spring AI, LangChain, LangChain4j, Rust agents, Java agents, Python agents, computer vision, games, classical ML, deep learning, UI apps, and full-stack systems.

The useful lessons are not about hype. They are about what kept breaking, what kept working, and what changed after building the same kind of thing many times. The most valuable ones are counterintuitive: they only became visible after the same mistake appeared across dozens of unrelated projects.

1. A POC is only useful when it removes one real uncertainty.

If the project tries to prove the model, the UI, the framework, the deployment, and the product idea at the same time, the result is noise. The best POCs made one decision easier. Before starting, name the single question the POC must answer. If you cannot name it, you are not ready to build.

2. The first version proves possibility. The second version exposes truth.

The first build often works because of lucky scope, cached context, or a narrow happy path. Rebuilding the same idea with less code shows what was essential. The gap between the two versions is where the real learning lives.

3. The environment breaks more often than the model.

Port collisions between sibling POCs, stale containers still serving old assets, sandboxes that block localhost, SELinux refusing a bind mount, a webcam the browser grabs at the wrong moment. Most lost hours were not the model being wrong. They were the machine not being in the state I assumed. Treat the runtime as a suspect first.

4. Reproducibility is the actual deliverable.

A POC nobody else can run proves nothing, and after a month that includes me. A run script, a stop script, a test script, and honest setup notes are worth more than a clever result. If the only place it works is the terminal it was born in, it did not really work.

5. Dependency drift is silent, and it is dated.

A pip wheel quietly drops a submodule, a framework renames a package, a model endpoint changes its defaults. The install that worked last month is not the install today. When something that worked stops working and the code never changed, suspect the versions before the logic. Trust the version in front of you, not the one you remember.

6. The model is rarely the whole bottleneck.

Bad context, unclear acceptance criteria, slow tools, missing state, weak retrieval, and poor UI often look like model failure. Switching models can hide the real issue instead of fixing it. Change the model last, not first.

7. Agents need smaller jobs than people expect.

An agent with a broad mission wastes time deciding what work means. An agent with one goal, one tool boundary, and one visible success condition behaves much better. The instinct to give it more autonomy is almost always wrong.

8. A tool call is a contract, not a convenience.

If the tool has unclear inputs, vague output, hidden side effects, or no error shape, the agent becomes unreliable. Good tool design matters more than clever prompting. The agent can only be as precise as the worst tool you hand it.

9. More context can make the answer worse.

Huge prompts create attention debt. Stuffing everything in dilutes the few tokens that mattered. The better pattern is small context, strong constraints, retrieval over inclusion, and a fast way to catch misses.

10. The best agent harness is boring.

Run scripts, test scripts, logs, screenshots, fixtures, and stable commands matter more than advanced orchestration. If the agent cannot run and verify the work, the output is just a claim. Boring infrastructure is what makes ambitious agents possible.

11. "Done" is the most dangerous word an agent says.

Agents report success over silent skips, missing steps, and tests that never ran. Completed is a lie if anything was quietly dropped. Design the loop so failure is loud and so success has to be shown, not asserted. Never accept a claim you cannot see verified.

12. Verify with your eyes, not only the logs.

A whole class of bugs never reaches the log: an emoji that eats the space after it, a WebGL canvas that silently falls back to blank, a layout that renders but is unusable. Screenshots and a real screen caught what text output swore was fine. For anything visual, seeing it is the test.

13. UI reveals problems backend POCs hide.

Latency, loading states, empty states, hallucinated features, confusing flows, and missing recovery paths become obvious when the idea has a screen. A visible app is a strong test of whether the idea is actually usable, not just technically possible.

14. The happy path is a liar.

The idea looks done when the ideal input works. Usefulness lives in the empty state, the error state, the timeout, the malformed response, and the way back from each. A POC that only handles the good case has tested the least interesting part of the problem.

15. RAG fails quietly.

The system can retrieve plausible but wrong context and still sound confident. Chunking, metadata, ranking, source quality, and refusal behavior matter more than which vector database is used. A confident wrong answer is worse than a visible miss.

16. Abstention is a feature you have to design.

A system that says "I do not know" or refuses low-quality input beats one that always answers. Models default to answering. Making them stop, ask, or decline is deliberate engineering, and it is usually the difference between a useful tool and a convincing liar.

17. Deterministic guardrails beat asking the model to be careful.

A validator, a schema check, a regex, a pre-flight assertion removes a failure mode that a polite instruction only reduces. Whenever correctness matters, put it in code the model cannot talk its way around.

18. Cost and latency are design inputs, not afterthoughts.

How many calls, how large, how fast, and how expensive shape what the product can even be. A flow that needs five model round-trips per action is a different product than one that needs one. Discover this at the start, because it decides the architecture.

19. Local models change experimentation behavior.

When running a model locally is cheap and private, the threshold for trying strange ideas drops. Local models are not always stronger, but they make exploration faster and less fragile, and cheap exploration is where most of the real lessons came from.

20. Frameworks age faster than patterns.

Specific agent and AI frameworks changed constantly. Tool calling, state machines, evals, retrieval discipline, isolation, human approval, and observability kept showing up as durable ideas. Invest in the patterns; treat the frameworks as replaceable.

21. The useful abstraction is often smaller than an agent.

A deterministic function, CLI wrapper, structured file format, narrow prompt, or pre-check can remove more risk than a larger autonomous workflow. Reach for the smallest thing that removes the uncertainty, not the most impressive one.

22. Computer vision and games are brutal quality tests.

They force timing, state transitions, visual clarity, input handling, and recovery under continuous pressure. Weak abstractions that survive a request-response POC collapse fast when the system has to react every frame.

23. Classical ML did not become obsolete.

Sklearn, numpy, pandas, scipy, PyTorch, TensorFlow, OpenCV, SHAP, NLP libraries, and simple algorithms still solve many problems better, cheaper, and more predictably than an LLM. The skill is choosing the right class of tool, not defaulting to the newest one.

24. A coding agent is only as good as its feedback loop.

Agents improve when they can inspect files, run tests, see failures, and make another pass. Without that loop, they mostly generate plausible text. The loop, not the model, is what turns generation into engineering.

25. Same task, different coding agent, different failure.

Some agents miss requirements. Some create better UI. Some handle permissions better. Some get stuck on setup. Running identical tasks across tools is the only honest benchmark, and it teaches more than any feature list or leaderboard.

26. Permissions are part of the design.

File access, shell access, browser access, network access, secrets, and write permissions shape what an agent can safely do. Security added later feels bolted on because it is. Decide the blast radius before the agent has one.

27. Observability must arrive earlier for AI systems.

With normal code, logs often explain what happened. With agents, you need prompts, tool calls, intermediate state, retries, latency, costs, and final outputs to understand a failure. If you cannot replay the decision, you cannot fix it.

28. The product flow decides how much intelligence is needed.

A crisp workflow may need one model call. A vague workflow may need planning, tools, memory, review, and still fail. Product clarity reduces AI complexity. Vagueness in the spec becomes unreliability in the system.

29. Prompt quality matters less than task shape.

A strong prompt cannot save a task with unclear data, no success criteria, and unsafe tools. A simple prompt works well when the surrounding system is precise. Fix the task shape before polishing the words.

30. POCs should be allowed to kill ideas.

A failed POC is useful when it reveals a bad assumption, an integration wall, a cost problem, or a reliability gap. Treating every POC as something that must become a product corrupts the learning. A cleanly killed idea is a successful POC.

31. Public work forces sharper thinking.

When the project is visible, it has to be easier to run, explain, and inspect. That pressure improves the work more than private notes do. Building in the open is a quality mechanism, not just marketing.

32. Write down what broke while it is still fresh.

The compounding asset across 500 POCs was not the code, it was the log of failures: the port that clashes, the wheel that drops a module, the mount that fails, the space an emoji eats. Recorded failures turn the same wall from a lost afternoon into a two-minute fix. Memory of what broke is worth more than memory of what worked.

33. Repetition builds better taste than one big project.

One POC teaches a tool. Ten teach a category. One hundred teach failure patterns. Five hundred teach when not to use AI at all, which is the most valuable judgment of the set.

34. The strongest lesson is still simple.

Build small, run it, compare it, inspect the failure, keep the useful pattern, write down what broke, and move on.
