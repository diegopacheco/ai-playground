import { serve } from "bun";
import { readFileSync } from "fs";
import { join } from "path";

const server = serve({
  port: 3000,
  async fetch(req) {
    const url = new URL(req.url);
    let filePath = url.pathname;
    
    // Serve index.html for root path
    if (filePath === "/") {
      filePath = "/index.html";
    }
    
    try {
      // Handle TypeScript/JSX files
      if (filePath.endsWith('.tsx') || filePath.endsWith('.ts')) {
        const build = await Bun.build({
          entrypoints: [join(process.cwd(), filePath.slice(1))],
          target: 'browser',
        });
        
        if (build.success && build.outputs[0]) {
          return new Response(await build.outputs[0].text(), {
            headers: { "Content-Type": "application/javascript" },
          });
        }
      }
      
      // Serve static files
      const file = Bun.file(join(process.cwd(), filePath.slice(1)));
      
      if (await file.exists()) {
        return new Response(file);
      }
      
      // Fallback to index.html for SPA routing
      const indexFile = Bun.file(join(process.cwd(), "index.html"));
      return new Response(indexFile);
      
    } catch (error) {
      console.error("Error serving file:", error);
      return new Response("Internal Server Error", { status: 500 });
    }
  },
});

console.log(`🎮 Rock Paper Scissors Game running at http://localhost:${server.port}`);
console.log("Press Ctrl+C to stop the server");