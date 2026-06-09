import OpenAI from "openai";
import { MAX_GUESSES, MAX_QUESTIONS, type Move, type Turn } from "./types";

let client: OpenAI | null = null;

function getClient(): OpenAI {
  if (!client) {
    client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  }
  return client;
}

const SYSTEM_PROMPT = `You are the GUESSER in a mind-reading party game.

The human is silently thinking of one concrete thing: a food, a place, a scene, a mood, a moment or a situation. Real examples: "a Brazilian hotdog", "a rainy day in Japan", "a hot summer afternoon", "the smell of fresh coffee".

YOUR JOB: figure out exactly what they are thinking.

RULES:
- You may ask the human anything, but they can ONLY answer with HOT or COLD.
  HOT means "yes / warm / you are close / on the right track".
  COLD means "no / cold / you are far off".
- Ask sharp questions that cut the space of possibilities in half.
- You may ask at most ${MAX_QUESTIONS} questions in the whole game. Once your questions run out you MUST guess.
- When you feel confident, make a GUESS of the exact thing.
- You only get ${MAX_GUESSES} guesses total for the whole game. Spend them wisely.
- A guess is only a win if it names the specific thing the human had in mind.
- Never repeat a question or guess you already made.
- Be playful, witty and concise. One sentence per move.

You MUST reply with a single minified JSON object, nothing else:
{"type":"question"|"guess","text":"<what you say to the human>","reasoning":"<short private note on your strategy>"}`;

export async function nextMove(turns: Turn[]): Promise<Move> {
  const questionsUsed = turns.filter((t) => t.type === "question").length;
  const guessesUsed = turns.filter((t) => t.type === "guess").length;
  const questionsRemaining = MAX_QUESTIONS - questionsUsed;
  const guessesRemaining = MAX_GUESSES - guessesUsed;
  const mustGuess = questionsRemaining <= 0;

  const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [
    { role: "system", content: SYSTEM_PROMPT },
  ];

  for (const turn of turns) {
    messages.push({
      role: "assistant",
      content: JSON.stringify({
        type: turn.type,
        text: turn.text,
        reasoning: turn.reasoning,
      }),
    });
    if (turn.response) {
      const label =
        turn.response === "CORRECT"
          ? "The human says: CORRECT! That is exactly it. You won."
          : turn.response === "WRONG"
            ? "The human says: WRONG, that is not it."
            : `The human reacts: ${turn.response}`;
      messages.push({ role: "user", content: label });
    }
  }

  const budget = `You have ${questionsRemaining} questions and ${guessesRemaining} guesses left.`;
  const order = mustGuess
    ? "You are out of questions — you MUST make a guess now, not a question."
    : "Make your next move.";

  messages.push({
    role: "user",
    content:
      turns.length === 0
        ? `The human is now thinking of their thing. ${budget} Make your first move.`
        : `${budget} ${order}`,
  });

  const completion = await getClient().chat.completions.create({
    model: "gpt-4o",
    temperature: 0.9,
    response_format: { type: "json_object" },
    messages,
  });

  const raw = completion.choices[0]?.message?.content ?? "{}";
  const parsed = JSON.parse(raw) as Partial<Move>;

  const type: Move["type"] =
    mustGuess || parsed.type === "guess" ? "guess" : "question";
  return {
    type,
    text: (parsed.text ?? "Hmm, let me think...").trim(),
    reasoning: (parsed.reasoning ?? "").trim(),
  };
}
