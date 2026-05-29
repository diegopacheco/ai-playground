import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";

const DEFAULT_PROMPTS = [
  {
    title: "Neon Tokyo Rain",
    text: "A cinematic shot gliding through a neon-lit Tokyo street at night, heavy rain reflecting pink and blue signs, a lone figure with an umbrella walking slowly, steam rising from the ground.",
  },
  {
    title: "Underwater Whale",
    text: "A majestic blue whale swimming through crystal clear turquoise water, sunbeams piercing the surface, schools of silver fish scattering, slow graceful camera movement following the whale.",
  },
  {
    title: "Desert Time-lapse",
    text: "A sweeping time-lapse over golden sand dunes at sunset, long shadows shifting, wind sculpting the sand, the sky transitioning from orange to deep purple with emerging stars.",
  },
  {
    title: "Cyberpunk Drone Chase",
    text: "A fast first-person drone flight chasing a flying car between towering futuristic skyscrapers, holographic billboards flickering, dramatic motion blur and lens flares.",
  },
  {
    title: "Cozy Forest Cabin",
    text: "A warm cozy wooden cabin in a snowy pine forest at dusk, smoke curling from the chimney, golden light glowing in the windows, soft snowflakes drifting down, gentle dolly-in shot.",
  },
];

async function startGeneration(prompt) {
  const res = await fetch("/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt }),
  });
  if (!res.ok) throw new Error((await res.json()).error || "Failed to start");
  return res.json();
}

async function fetchStatus(jobId) {
  const res = await fetch(`/api/status/${jobId}`);
  if (!res.ok) throw new Error("Failed to fetch status");
  return res.json();
}

export default function App() {
  const [prompt, setPrompt] = useState("");
  const [jobId, setJobId] = useState(null);

  const mutation = useMutation({
    mutationFn: startGeneration,
    onSuccess: (data) => setJobId(data.jobId),
  });

  const status = useQuery({
    queryKey: ["status", jobId],
    queryFn: () => fetchStatus(jobId),
    enabled: !!jobId,
    refetchInterval: (q) => {
      const s = q.state.data?.status;
      return s === "done" || s === "error" ? false : 4000;
    },
  });

  const isWorking = mutation.isPending || status.data?.status === "running";
  const videoUrl = status.data?.status === "done" ? status.data.videoUrl : null;
  const errorMsg = mutation.error?.message || status.data?.error;

  const submit = () => {
    if (!prompt.trim()) return;
    setJobId(null);
    mutation.mutate(prompt.trim());
  };

  return (
    <div className="app">
      <header className="header">
        <h1>Veo Video Studio</h1>
        <p>Generate cinematic video from text with Google Veo 3.1</p>
      </header>

      <section className="card">
        <label className="label">Pick a prompt or write your own</label>
        <div className="chips">
          {DEFAULT_PROMPTS.map((p) => (
            <button
              key={p.title}
              className="chip"
              disabled={isWorking}
              onClick={() => setPrompt(p.text)}
            >
              {p.title}
            </button>
          ))}
        </div>

        <textarea
          className="textarea"
          rows={5}
          placeholder="Describe the video you want to generate..."
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          disabled={isWorking}
        />

        <button className="generate" onClick={submit} disabled={isWorking || !prompt.trim()}>
          {isWorking ? "Generating..." : "Generate Video"}
        </button>
      </section>

      {isWorking && (
        <section className="card status">
          <div className="spinner" />
          <p>Veo is generating your video. This usually takes 1–3 minutes.</p>
        </section>
      )}

      {errorMsg && (
        <section className="card error">
          <strong>Error:</strong> {errorMsg}
        </section>
      )}

      {videoUrl && (
        <section className="card result">
          <h2>Your video is ready</h2>
          <video src={videoUrl} controls autoPlay loop className="video" />
          <a className="download" href={videoUrl} download="veo-video.mp4">
            Download MP4
          </a>
        </section>
      )}
    </div>
  );
}
