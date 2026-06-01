import index from "./index.html";

const server = Bun.serve({
  port: Number(process.env.PORT ?? 3000),
  routes: {
    "/": index,
  },
  development: true,
});

console.log(`3D Tetris listening on ${server.url}`);
