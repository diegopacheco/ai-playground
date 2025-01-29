### DeepSeek Model

* https://ollama.com/library/deepseek-r1
* https://github.com/deepseek-ai/DeepSeek-R1

### Run

Run ollama server running DeepSeek-R1 model locally

```bash
./run-ollama-server.sh
```

Run the NodeJS app using ollama api.

```bash
npm run run
```

### Results

```
❯ npm run run

> ollama-node@1.0.0 run
> node src/index.js

Request: {
  model: 'deepseek-r1:1.5b',
  messages: [ { role: 'user', content: 'Why is the sky blue?' } ]
}
Response: {
  model: 'deepseek-r1:1.5b',
  created_at: '2025-01-29T06:48:36.283139576Z',
  message: {
    role: 'assistant',
    content: '<think>\n' +
      '\n' +
      '</think>\n' +
      '\n' +
      "The color of the sky, also known as the atmospheric color, is primarily caused by Earth's curvature. As sunlight passes through Earth's atmosphere, it refracts and then reflects off the Earth's surface to create the red or pink colors we observe when the sun sets (red) or rises (pink). This phenomenon is a result of light waves having a range of wavelengths, with shorter wavelengths at the top of the spectrum being absorbed by gases like carbon dioxide in the atmosphere."
  },
  done_reason: 'stop',
  done: true,
  total_duration: 7050421690,
  load_duration: 1560777207,
  prompt_eval_count: 9,
  prompt_eval_duration: 178000000,
  eval_count: 100,
  eval_duration: 5310000000
}❯ npm run run

> ollama-node@1.0.0 run
> node src/index.js

Request: {
  model: 'deepseek-r1:1.5b',
  messages: [ { role: 'user', content: 'Why is the sky blue?' } ]
}
Response: {
  model: 'deepseek-r1:1.5b',
  created_at: '2025-01-29T06:48:36.283139576Z',
  message: {
    role: 'assistant',
    content: '<think>\n' +
      '\n' +
      '</think>\n' +
      '\n' +
      "The color of the sky, also known as the atmospheric color, is primarily caused by Earth's curvature. As sunlight passes through Earth's atmosphere, it refracts and then reflects off the Earth's surface to create the red or pink colors we observe when the sun sets (red) or rises (pink). This phenomenon is a result of light waves having a range of wavelengths, with shorter wavelengths at the top of the spectrum being absorbed by gases like carbon dioxide in the atmosphere."
  },
  done_reason: 'stop',
  done: true,
  total_duration: 7050421690,
  load_duration: 1560777207,
  prompt_eval_count: 9,
  prompt_eval_duration: 178000000,
  eval_count: 100,
  eval_duration: 5310000000
}
```