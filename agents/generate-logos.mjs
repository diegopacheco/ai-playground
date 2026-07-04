import { execFileSync } from "node:child_process"
import { existsSync, mkdirSync, readFileSync, rmSync, unlinkSync } from "node:fs"
import { basename, dirname, extname, join } from "node:path"
import { fileURLToPath } from "node:url"

const root = dirname(fileURLToPath(import.meta.url))
const source = readFileSync(join(root, "landing-page.html"), "utf8")
const projects = JSON.parse(source.match(/const projects=(\[.*?\]);/s)[1])
const output = join(root, "logos")
const args = new Map(process.argv.slice(2).map(value => value.split("=")))
const from = Number(args.get("--from") ?? 1)
const to = Number(args.get("--to") ?? projects.length)
const force = args.has("--force")
const model = args.get("--model") ?? "flux_1_schnell_q5p.ckpt"
const steps = args.get("--steps") ?? "4"
const size = args.get("--size") ?? "512"
const motifs = {
  "Agent Systems": "autonomous core and connected circuit node",
  "Agent Skills": "precision tool and bright capability spark",
  "Evaluations": "balanced scales and analytical signal",
  "Harnesses": "structured rails surrounding a controlled core",
  "Infrastructure": "stacked platform and resilient network",
  "MCP": "modular plug and connected context nodes",
  "Memory": "layered memory crystal and circular recall path",
  "Multi-Agent": "three coordinated nodes orbiting one shared goal",
  "Observability": "clear lens over a measured signal",
  "Protocols": "interlocking links and verified data path",
  "Rust & Scala": "strong angular systems symbol and precise gear",
  "Safety": "protective shield around a controlled circuit",
  "Tools & Research": "focused instrument and discovery beam"
}
const colors = ["navy blue and coral", "deep teal and amber", "indigo and cyan", "forest green and gold", "royal blue and magenta", "charcoal and vermilion"]
const ownedLogos = {
  1: "https://raw.githubusercontent.com/diegopacheco/codex-poc/main/logo-app.png",
  2: "https://raw.githubusercontent.com/diegopacheco/google-jules-poc/main/logo-app.png",
  3: "https://raw.githubusercontent.com/diegopacheco/claude-code-poc/main/logo-app.png",
  6: "https://raw.githubusercontent.com/diegopacheco/sketch-dev-poc/main/logo-app.png",
  7: "https://raw.githubusercontent.com/diegopacheco/augmentcode-poc/main/logo-app.png",
  8: "https://raw.githubusercontent.com/diegopacheco/opencode-poc/main/logo-app.png",
  9: "https://raw.githubusercontent.com/diegopacheco/cursor-agent-gpt5-poc/main/logo-app.png",
  10: join(root, "..", "..", "aws-kiro-poc", "logo-app.png"),
  68: "https://raw.githubusercontent.com/diegopacheco/Smith/main/smith-logo.png",
  73: join(root, "..", "pocs", "local-agent-orama", "local-agent-orama-logo-gemin-3-nana-banana-pro.png"),
  74: join(root, "..", "..", "local-agent-rust-llama3", "llr3-logo.png"),
  75: join(root, "..", "..", "multi-agent-verse", "transparent-logo.png"),
  76: join(root, "..", "..", "prompt-2-k8s-agent", "logo-treansparent.png"),
  110: join(root, "..", "pocs", "k8s-sre-agent-operator", "logo.png"),
  150: join(root, "..", "pocs", "tambo-generative-ui-app-fun", "public", "Octo-Icon.svg"),
  152: join(root, "..", "pocs", "claude-opus-4.5-ai-summarizator", "logo-ras.png"),
  159: join(root, "..", "pocs", "ai-guessing-game", "app", "icon.svg")
}

mkdirSync(output, { recursive: true })

const slug = value => value.toLowerCase().normalize("NFKD").replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 64)

const normalizeLogo = (source, target, id) => {
  let input = source
  const downloaded = join(output, `.owned-${id}.png`)
  if (source.startsWith("https://")) {
    execFileSync("curl", ["-fsSL", source, "-o", downloaded])
    input = downloaded
  }
  let rendered
  if (extname(input).toLowerCase() === ".svg") {
    const renderDir = join(output, `.owned-${id}`)
    mkdirSync(renderDir, { recursive: true })
    execFileSync("qlmanage", ["-t", "-s", "480", "-o", renderDir, input], { stdio: "ignore" })
    rendered = join(renderDir, `${basename(input)}.png`)
    input = rendered
  }
  const scaled = join(output, `.scaled-${id}.png`)
  execFileSync("sips", ["-Z", "480", input, "--out", scaled], { stdio: "ignore" })
  execFileSync("sips", ["-p", "528", "528", "--padColor", "FFFFFF", scaled, "--out", target], { stdio: "ignore" })
  unlinkSync(scaled)
  if (source.startsWith("https://")) unlinkSync(downloaded)
  if (rendered) rmSync(dirname(rendered), { recursive: true, force: true })
}

for (const project of projects.filter(item => item.id >= from && item.id <= to)) {
  const name = `${String(project.id).padStart(3, "0")}-${slug(project.name)}.png`
  const target = join(output, name)
  if (ownedLogos[project.id]) {
    console.log(`[${project.id}/${projects.length}] normalizing owned logo ${name}`)
    normalizeLogo(ownedLogos[project.id], target, project.id)
    continue
  }
  if (!force && existsSync(target)) {
    console.log(`[${project.id}/${projects.length}] skipped ${name}`)
    continue
  }
  const raw = join(output, `.${name}`)
  const motif = motifs[project.category] ?? "distinctive abstract technology symbol"
  const palette = colors[(project.id - 1) % colors.length]
  const prompt = `Minimal vector logo mark for ${project.name}. Concept: ${project.description} Visual motif: ${motif}. Centered single symbol, distinctive silhouette, bold geometric shapes, crisp clean edges, ${palette} palette, pure white background, generous empty margin, flat 2D icon design, professional technology brand identity, no text, no letters, no words, no watermark, no border, no shadow, no gradient.`
  console.log(`[${project.id}/${projects.length}] generating ${name}`)
  execFileSync("draw-things-cli", ["generate", "--model", model, "--prompt", prompt, "--negative-prompt", "text, letters, words, typography, watermark, signature, border, frame, mockup, photograph, realistic, shadow, gradient, clutter, busy background, multiple logos", "--steps", steps, "--cfg", "1", "--width", size, "--height", size, "--seed", String(1000 + project.id), "--output", raw, "--disable-preview", "--offline"], { stdio: ["ignore", "ignore", "inherit"] })
  execFileSync("sips", ["-z", "528", "528", raw, "--out", target], { stdio: "ignore" })
  unlinkSync(raw)
  console.log(`[${project.id}/${projects.length}] wrote ${target}`)
}
