const PROMPT =
  "What is this object? Answer in one short phrase, include the brand if recognizable. If unsure, give your best guess.";

const SAMPLE_BASE64 =
  "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAARCAABAAEDASIAAhEBAxEB/8QAFAABAAAAAAAAAAAAAAAAAAAAA//EABQQAQAAAAAAAAAAAAAAAAAAAAD/xAAUAQEAAAAAAAAAAAAAAAAAAAAA/8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAwDAQACEQMRAD8AfwD/2Q==";

async function main() {
  const apiKey = process.env.EXPO_PUBLIC_OPENAI_API_KEY;
  if (!apiKey) {
    console.error("FAIL: EXPO_PUBLIC_OPENAI_API_KEY is not set");
    process.exit(1);
  }

  const res = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: "gpt-4o",
      max_tokens: 100,
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: PROMPT },
            {
              type: "image_url",
              image_url: { url: `data:image/jpeg;base64,${SAMPLE_BASE64}` },
            },
          ],
        },
      ],
    }),
  });

  if (!res.ok) {
    const body = await res.text();
    console.error(`FAIL: OpenAI ${res.status}: ${body}`);
    process.exit(1);
  }

  const data = await res.json();
  const label = data?.choices?.[0]?.message?.content?.trim();
  if (!label) {
    console.error("FAIL: no label returned");
    process.exit(1);
  }

  console.log(`PASS: model returned label -> "${label}"`);
}

main().catch((e) => {
  console.error(`FAIL: ${e.message}`);
  process.exit(1);
});
