import { execFile } from 'node:child_process';

function runClaudeAgent(role: string, prompt: string): Promise<string> {
  const fullPrompt = `${role}\n\n${prompt}`;
  return new Promise((resolve, reject) => {
    execFile(
      'claude',
      ['-p', fullPrompt],
      { maxBuffer: 10 * 1024 * 1024, timeout: 110_000 },
      (error, stdout, stderr) => {
        if (error) {
          reject(new Error(stderr.trim() || error.message));
          return;
        }
        const output = stdout.trim();
        if (output.length === 0) {
          reject(new Error('claude returned empty output'));
          return;
        }
        resolve(output);
      },
    );
  });
}

export async function researchTopic(topic: string): Promise<string> {
  return runClaudeAgent(
    'You are a research agent.',
    `List 5 concise, factual bullet points about the topic "${topic}". Output only the bullet points.`,
  );
}

export async function writeDraft(topic: string, research: string): Promise<string> {
  return runClaudeAgent(
    'You are a writing agent.',
    `Using the research below, write a clear 150-word article about "${topic}". Output only the article.\n\nResearch:\n${research}`,
  );
}

export async function critiqueDraft(draft: string): Promise<string> {
  return runClaudeAgent(
    'You are a critique agent.',
    `Give 3 specific, actionable suggestions to improve the draft below. Output only the numbered suggestions.\n\nDraft:\n${draft}`,
  );
}

export async function editDraft(draft: string, critique: string): Promise<string> {
  return runClaudeAgent(
    'You are an editor agent.',
    `Rewrite the draft below applying every suggestion. Output only the final article.\n\nDraft:\n${draft}\n\nSuggestions:\n${critique}`,
  );
}
