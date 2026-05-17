import { serveDir } from "@std/http/file-server";

const port = 8000;

Deno.serve({ port }, (req) =>
  serveDir(req, { fsRoot: "./public", showIndex: true })
);

console.log(`Memory game running at http://localhost:${port}`);
