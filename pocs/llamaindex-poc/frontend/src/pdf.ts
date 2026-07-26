import * as pdfjs from "pdfjs-dist";
import workerSrc from "pdfjs-dist/build/pdf.worker.min.mjs?url";

pdfjs.GlobalWorkerOptions.workerSrc = workerSrc;

export interface RenderedPage {
  pages: number;
  width: number;
  height: number;
}

export async function renderPage(
  url: string,
  pageNumber: number,
  scale: number,
  canvas: HTMLCanvasElement,
): Promise<RenderedPage> {
  const document = await pdfjs.getDocument({ url }).promise;
  const target = Math.min(Math.max(pageNumber, 1), document.numPages);
  const page = await document.getPage(target);
  const viewport = page.getViewport({ scale });
  const context = canvas.getContext("2d");
  if (!context) throw new Error("canvas 2d context unavailable");

  canvas.width = Math.floor(viewport.width);
  canvas.height = Math.floor(viewport.height);
  await page.render({ canvas, canvasContext: context, viewport }).promise;

  return { pages: document.numPages, width: canvas.width, height: canvas.height };
}
