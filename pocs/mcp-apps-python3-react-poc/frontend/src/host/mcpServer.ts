import { Client, StreamableHTTPClientTransport } from "@modelcontextprotocol/client";
import { getToolUiResourceUri, RESOURCE_MIME_TYPE } from "@modelcontextprotocol/ext-apps/app-bridge";

export const HOST_INFO = { name: "calculator-host", version: "0.1.0" };
export const UI_EXTENSION = "io.modelcontextprotocol/ui";

export type ServerView = {
  client: Client;
  toolName: string;
  html: string;
};

export async function connectServer(endpoint: URL): Promise<Client> {
  const client = new Client(HOST_INFO, {
    capabilities: { extensions: { [UI_EXTENSION]: { mimeTypes: [RESOURCE_MIME_TYPE] } } },
  });
  await client.connect(new StreamableHTTPClientTransport(endpoint));
  return client;
}

export async function loadView(client: Client): Promise<ServerView> {
  const { tools } = await client.listTools();
  const tool = tools.find((candidate) => getToolUiResourceUri(candidate));
  const uri = tool ? getToolUiResourceUri(tool) : undefined;
  if (!tool || !uri) throw new Error("server exposes no tool with a UI resource");
  const { contents } = await client.readResource({ uri });
  const page = contents.find((content) => content.mimeType === RESOURCE_MIME_TYPE && "text" in content);
  if (!page || !("text" in page)) throw new Error(`resource ${uri} has no ${RESOURCE_MIME_TYPE} content`);
  return { client, toolName: tool.name, html: page.text };
}
