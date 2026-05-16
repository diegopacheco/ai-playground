#!/usr/bin/env node
import OpenAI from "openai";

const baseURL = process.env.AGENTGATEWAY_URL || "http://localhost:8080/v1";

const client = new OpenAI({
  baseURL,
  apiKey: "unused-handled-by-agentgateway",
});

const prompt = process.argv.slice(2).join(" ").trim();
if (!prompt) {
  console.error("usage: chat.js <prompt>");
  process.exit(1);
}

const res = await client.chat.completions.create({
  model: "gpt-4o-mini",
  messages: [{ role: "user", content: prompt }],
});

console.log(res.choices[0].message.content);
