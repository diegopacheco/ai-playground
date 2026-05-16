#!/usr/bin/env node
import http from "node:http";

const UPSTREAM = process.env.UPSTREAM || "http://localhost:8080";
const PORT = Number(process.env.PORT || 8090);
const STRIP = new Set([
  "reasoning",
  "reasoning_effort",
  "thinking",
  "anthropic_beta",
  "metadata",
]);

const upstream = new URL(UPSTREAM);

const server = http.createServer(async (req, res) => {
  const chunks = [];
  for await (const c of req) chunks.push(c);
  const raw = Buffer.concat(chunks);

  let body = raw;
  const ct = req.headers["content-type"] || "";
  if (ct.includes("application/json") && raw.length > 0) {
    try {
      const j = JSON.parse(raw.toString("utf8"));
      for (const k of STRIP) delete j[k];
      body = Buffer.from(JSON.stringify(j));
    } catch {}
  }

  const headers = { ...req.headers, host: upstream.host };
  headers["content-length"] = String(body.length);

  const proxyReq = http.request(
    {
      hostname: upstream.hostname,
      port: upstream.port || 80,
      path: req.url,
      method: req.method,
      headers,
    },
    (upRes) => {
      res.writeHead(upRes.statusCode || 502, upRes.headers);
      upRes.pipe(res);
    },
  );
  proxyReq.on("error", (e) => {
    res.statusCode = 502;
    res.end(String(e));
  });
  proxyReq.end(body);
});

server.listen(PORT, () => {
  console.log(`strip-proxy on :${PORT} -> ${UPSTREAM}`);
});
