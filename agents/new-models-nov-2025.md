# New models - November 2025

Made some simple benchmark using latest version of VSCode Copilot with:
* Claude Opus 4.5
* GPT-5.1-Codex
* Gemini 3 Pro
* Grok Fast 1 Code

## Prompt 

```
create a rock, paper, scizor game using react and html, use browser local store for storage.
```

## Results

Claude Opus 4.5 <br/>
<img src="models-nov-2025/Rock Paper Scissors - opus 4.5 - result.png" width="600"/>

Full POC Code: [https://github.com/diegopacheco/ai-playground/tree/main/pocs/copilot-opus-4.5-poc](https://github.com/diegopacheco/ai-playground/tree/main/pocs/copilot-opus-4.5-poc)

GPT 5.1 Codex <br/>
<img src="models-nov-2025/Rock Paper Scissors - Grok-Fast-1 - result.png" width="600"/>

Full POC Code: [https://github.com/diegopacheco/ai-playground/tree/main/pocs/codex-5.1-code-poc](https://github.com/diegopacheco/ai-playground/tree/main/pocs/codex-5.1-code-poc)

Gemini 3 Pro <br/>
<img src="models-nov-2025/Rock Paper Scissors - Gemini 3 PRO - result.png" width="600"/>

Full POC Code: [https://github.com/diegopacheco/ai-playground/tree/main/pocs/gemini-3-poc](https://github.com/diegopacheco/ai-playground/tree/main/pocs/gemini-3-poc)

Grok Fast 1 Code <br/>
<img src="models-nov-2025/Rock Paper Scissors - Grok-Fast-1 - result.png" width="600"/>

Full Code: [https://github.com/diegopacheco/ai-playground/tree/main/pocs/grok-code-fast-1-poc](https://github.com/diegopacheco/ai-playground/tree/main/pocs/grok-code-fast-1-poc)

## analysis

* All modes was alble to do with 1 shot.
* GPT-5.1-Codex got stuck and I had to interrupt it.
* GPT-5.1-Codex I had to ask to create a run.sh. 
* GPT-5.1-Codex used React but as a remote script import in a html file :-) 
* Copilot still does comment no matter how many ignore files I have :( No matter the model.
* Claude Opus 4.5 was impressive.