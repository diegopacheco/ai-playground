import { Client } from "@modelcontextprotocol/client";
import { SkillsClient } from "@modelcontextprotocol/client/ext/skills";
import { StdioClientTransport } from "@modelcontextprotocol/client/stdio";
import { CLIENT_CAPABILITIES_META_KEY_V2, TASKS_EXTENSION_ID_V2 } from "@modelcontextprotocol/ext-tasks/core/v2";
import { RawDispatcher } from "./rawRpc.ts";
import { TaskClient } from "./taskClient.ts";

export type Connection = {
  readonly client: Client;
  readonly skills: SkillsClient;
  readonly tasks: TaskClient;
  readonly close: () => Promise<void>;
};

function processEnv(): Record<string, string> {
  return Object.fromEntries(Object.entries(Bun.env).filter((e): e is [string, string] => e[1] !== undefined));
}

const CLIENT_INFO = { name: "market-research-client", version: "1.0.0" } as const;
const CAPABILITIES = { extensions: { [TASKS_EXTENSION_ID_V2]: {} } } as const;

export async function connect(serverEntry: string, env: Readonly<Record<string, string>> = {}): Promise<Connection> {
  const transport = new StdioClientTransport({ command: "bun", args: [serverEntry], env: { ...processEnv(), ...env }, stderr: "ignore" });
  const client = new Client(CLIENT_INFO, { capabilities: CAPABILITIES, versionNegotiation: { mode: "auto" } });
  await client.connect(transport);
  const protocolVersion = client.getNegotiatedProtocolVersion();
  if (!protocolVersion) throw new Error("protocol version was not negotiated");
  const envelope = {
    "io.modelcontextprotocol/protocolVersion": protocolVersion,
    "io.modelcontextprotocol/clientInfo": CLIENT_INFO,
    [CLIENT_CAPABILITIES_META_KEY_V2]: CAPABILITIES,
  };
  const tasks = new TaskClient(new RawDispatcher(transport, envelope), (ms) => Bun.sleep(ms));
  return { client, skills: new SkillsClient(client), tasks, close: () => client.close() };
}
