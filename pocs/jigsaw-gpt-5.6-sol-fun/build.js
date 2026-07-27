const fs = require("node:fs")
const path = require("node:path")

const source = path.join(__dirname, "public")
const destination = path.join(__dirname, "dist")

fs.rmSync(destination, { recursive: true, force: true })
fs.cpSync(source, path.join(destination, "client"), { recursive: true })
fs.mkdirSync(path.join(destination, "server"), { recursive: true })
fs.mkdirSync(path.join(destination, ".openai"), { recursive: true })
fs.copyFileSync(path.join(__dirname, "worker.mjs"), path.join(destination, "server", "index.js"))
fs.copyFileSync(path.join(__dirname, ".openai", "hosting.json"), path.join(destination, ".openai", "hosting.json"))
