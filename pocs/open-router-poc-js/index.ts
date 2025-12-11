import { OpenRouter } from '@openrouter/sdk';

const openRouter = new OpenRouter({
  apiKey: process.env.OPENROUTER_API_KEY,
  httpReferer: '127.0.0.1',
  xTitle: 'Open-Router-POC',
});

const completion = await openRouter.chat.send({
  model: 'amazon/nova-2-lite-v1:free',
  messages: [
    {
      role: 'user',
      content: 'What is the meaning of life?',
    },
  ],
  stream: false,
});

console.log(completion.choices[0].message.content);