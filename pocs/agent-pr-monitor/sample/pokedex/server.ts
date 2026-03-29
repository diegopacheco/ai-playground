const port = 4200;
const dir = "./dist/pokedex/browser";

Deno.serve({ port }, async (req: Request) => {
  const url = new URL(req.url);
  let path = url.pathname === "/" ? "/index.html" : url.pathname;
  try {
    const file = await Deno.readFile(dir + path);
    const ext = path.split(".").pop() || "";
    const types: Record<string, string> = {
      html: "text/html", css: "text/css", js: "application/javascript",
      json: "application/json", png: "image/png", jpg: "image/jpeg",
      svg: "image/svg+xml", ico: "image/x-icon", woff2: "font/woff2",
    };
    return new Response(file, {
      headers: { "content-type": types[ext] || "application/octet-stream" },
    });
  } catch {
    const index = await Deno.readFile(dir + "/index.html");
    return new Response(index, { headers: { "content-type": "text/html" } });
  }
});
