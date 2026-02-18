import * as readline from "readline";

const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;

if (!OPENROUTER_API_KEY) {
  console.error("Error: OPENROUTER_API_KEY environment variable is not set.");
  process.exit(1);
}

async function getCityCuriosities(city: string): Promise<string> {
  const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${OPENROUTER_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      models: [
        "google/gemma-3-12b-it:free",
        "mistralai/mistral-small-3.1-24b-instruct:free",
        "deepseek/deepseek-r1-0528:free",
      ],
      messages: [
        {
          role: "user",
          content: `Tell me exactly 3 interesting curiosities about the city of ${city}. Number each curiosity (1, 2, 3).`,
        },
      ],
    }),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`OpenRouter API error ${response.status}: ${error}`);
  }

  const data = await response.json() as {
    choices: Array<{ message: { content: string } }>;
  };

  return data.choices[0].message.content;
}

async function main(): Promise<void> {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const city = await new Promise<string>((resolve) => {
    rl.question("Enter a city name: ", (answer) => {
      rl.close();
      resolve(answer.trim());
    });
  });

  if (!city) {
    console.error("Error: City name cannot be empty.");
    process.exit(1);
  }

  console.log(`\nFetching curiosities about ${city}...\n`);

  const curiosities = await getCityCuriosities(city);
  console.log(curiosities);
}

main().catch((err: Error) => {
  console.error("Error:", err.message);
  process.exit(1);
});
