# Frontend Tools Models Coding Agents

I played with a bunch of coding agents, giving them the same prompt, also reading the code and seeing what they was doing. 

## The Prompt

```
build a game called "Who want's to be a Vibe Coder?" as a satire of "who wants to be a milionare". Dont use
comments never ever, build this with react, dont use backend or redux, save it all on the browser local store. Make sure every question has 4 options, 3 wrong 1 right, every time the question is right - render a congratulations annimations with confetti, ask 10 questions give 21s for each question for the user to anwser, to help the user there is 3 options(buttones). The user can: A: skip (the user can skip 2 times and get new questions - but he needs anwseer 10 questions no matter what). B: Vibe code them a ramdon rolete
happens and user might win or losse (3) win 10 more seconds just 1 time. The questions must be about Distributed Systems, Design Patterns, OOP, FP, Data Srtructures, Algorithims, Cloud Computing on AWS, DevOps, Data Engineering, Frontend Engineering and about weired langs like Haskell, Rust, Zig, Nim, Clojure, Emojicode and TypeScript. Create a run.sh to run the app.
```

## Contenders

* [OpenAI Codex and GPT-5.1-MAX](https://github.com/diegopacheco/ai-playground/tree/main/pocs/codex-gpt-fun)
* [Google Gemini CLI and 2.5 PRO](https://github.com/diegopacheco/ai-playground/tree/main/pocs/gemini-2.5-pro-cli-fun/vibe-coder-game)
* [Github/MS Copilot and Claude-Sonnet-4.5](https://github.com/diegopacheco/ai-playground/tree/main/pocs/copilot-cli-sonnet-4-5-fun/vibe-coder-game)
* [Anthropic Claude Code / Sonner 4.5]()

## Analysis / Results 

* Codex was the first to finish.
* Copilot was second.
* Gemini Last.
* Claude I could not run (out of tokens - will run another day).
* Codex UI was the cool, however it game me a single HTML and I asked for React :-) 
* Codex Got it one shot(but without React).
* Copilot took a while and miss 1 functionality the 10s button.
* Gemini took a while and miss the 10s button and skip button.
* Copilot and codex was decent cli tools
* Gemini was bad, unless I allow the YOLO mode I could not use the tool, had lots of bugs.
* Gemini UI was ugly.
* Copilot UI and code was great.
* 