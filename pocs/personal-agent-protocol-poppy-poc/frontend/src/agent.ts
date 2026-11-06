import { useQuery } from "@tanstack/react-query";
import { PoppyClient } from "./poppy/client";
import { clientMetadata, createIdentity } from "./poppy/identity";
import type { PoppyDocument } from "./poppy/types";

export const COMPANY_URL = __COMPANY_URL__;
export const AGENT_NAME = "atlas";

export type Agent = { client: PoppyClient; document: PoppyDocument };

async function bootstrap(): Promise<Agent> {
  const clientId = `${COMPANY_URL}/agent-host/${AGENT_NAME}/agent.json`;
  const identity = await createIdentity(clientId);
  const discovery = new PoppyClient(identity, COMPANY_URL);
  const document = await discovery.call<PoppyDocument>(`${COMPANY_URL}/.well-known/poppy.json`, { session: false });
  await discovery.call(clientId, { method: "PUT", body: clientMetadata(identity), session: false });
  return { client: discovery, document };
}

export function useAgent() {
  return useQuery({ queryKey: ["agent"], queryFn: bootstrap, staleTime: Infinity, retry: 1 });
}

export function conversationEndpoint(document: PoppyDocument): string {
  const entry = document.agent?.protocols.find((protocol) => protocol.type === "poppy");
  if (!entry) throw new Error("the Company has no poppy conversation endpoint");
  return entry.endpoint;
}

export function apiBase(document: PoppyDocument): string {
  const api = document.apis?.find((item) => item.type === "openapi");
  if (!api) throw new Error("the Company lists no OpenAPI");
  return new URL(api.url).origin;
}
