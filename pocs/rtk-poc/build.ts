import * as esbuild from "esbuild";
import { denoPlugins } from "esbuild_deno_loader";

await esbuild.build({
  plugins: [...denoPlugins({ configPath: `${Deno.cwd()}/deno.json` })],
  entryPoints: ["./src/main.tsx"],
  bundle: true,
  outfile: "./public/bundle.js",
  format: "esm",
  jsx: "automatic",
  jsxImportSource: "https://esm.sh/react@18.3.1",
  loader: { ".tsx": "tsx", ".ts": "ts" },
  minify: false,
  sourcemap: true,
});

esbuild.stop();
