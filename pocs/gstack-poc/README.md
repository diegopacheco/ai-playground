# gstack

https://github.com/garrytan/gstack

## Install

```
git clone --single-branch --depth 1 https://github.com/garrytan/gstack.git ~/.claude/skills/gstack && cd ~/.claude/skills/gstack && ./setup
```

## Experience Notes

* Using claude code and opus 4.7 with 1M ctx window.
* Installation was easy and smoth
* /office-hours ask you a lot of questions
* The skill is way too big and blow the context window: "⚠ Large CLAUDE.md will impact performance (48.5k chars > 40.0k)"
* Just to desgin a md file took 14% of my subscription
* So the second command /plan-eng-review just keep trying to convince you to reduce scope
* /plan-eng-review was also pushing to consider SaaS to be used (Fly.io or Railway) - this is really a startup mentality.
* Skill sugest you run codex
* /design-shotgun it's interesting - it give you multiple options to chose.
* Skills are asking for open ai API KEY all the time - kind of anoying.
* Skill say why: 
```
⏺ Honest answer: Claude doesn't generate raster images. I can SEE images (vision), describe them, write code that renders images
  (HTML/SVG/canvas), but I can't synthesize a PNG/JPEG from a prompt the way DALL-E / Imagen / gpt-image-1 do. Anthropic has no public
  image-generation API. The gstack designer wraps OpenAI's image API, which is why it needs an OpenAI key.
```