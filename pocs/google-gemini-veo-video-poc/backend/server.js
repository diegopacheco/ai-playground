import { GoogleGenAI } from "@google/genai";
import express from "express";
import { randomUUID } from "node:crypto";
import { mkdirSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const VIDEOS_DIR = join(__dirname, "videos");
const PORT = process.env.PORT || 3001;
const MODEL = "veo-3.1-generate-preview";

if (!process.env.GEMINI_API_KEY) {
  console.error("GEMINI_API_KEY env var is required");
  process.exit(1);
}

if (!existsSync(VIDEOS_DIR)) mkdirSync(VIDEOS_DIR, { recursive: true });

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
const jobs = new Map();

async function runJob(jobId, prompt) {
  const job = jobs.get(jobId);
  try {
    let operation = await ai.models.generateVideos({ model: MODEL, prompt });
    while (!operation.done) {
      await new Promise((r) => setTimeout(r, 10000));
      operation = await ai.operations.getVideosOperation({ operation });
    }
    const generated = operation.response?.generatedVideos?.[0];
    if (!generated?.video) throw new Error("No video returned by the API");
    const fileName = `${jobId}.mp4`;
    await ai.files.download({
      file: generated.video,
      downloadPath: join(VIDEOS_DIR, fileName),
    });
    job.status = "done";
    job.fileName = fileName;
  } catch (err) {
    job.status = "error";
    job.error = err?.message || String(err);
    console.error(`Job ${jobId} failed:`, job.error);
  }
}

const app = express();
app.use(express.json());

app.post("/api/generate", (req, res) => {
  const prompt = (req.body?.prompt || "").trim();
  if (!prompt) return res.status(400).json({ error: "prompt is required" });
  const jobId = randomUUID();
  jobs.set(jobId, { status: "running", prompt });
  runJob(jobId, prompt);
  res.json({ jobId, status: "running" });
});

app.get("/api/status/:jobId", (req, res) => {
  const job = jobs.get(req.params.jobId);
  if (!job) return res.status(404).json({ error: "job not found" });
  res.json({
    status: job.status,
    error: job.error,
    videoUrl: job.status === "done" ? `/api/video/${req.params.jobId}` : null,
  });
});

app.get("/api/video/:jobId", (req, res) => {
  const job = jobs.get(req.params.jobId);
  if (!job || job.status !== "done") return res.status(404).end();
  res.sendFile(join(VIDEOS_DIR, job.fileName));
});

app.listen(PORT, () => console.log(`Backend listening on http://localhost:${PORT}`));
