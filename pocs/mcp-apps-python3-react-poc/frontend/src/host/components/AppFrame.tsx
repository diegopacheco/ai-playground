import { useEffect, useRef, useState } from "react";
import { AppBridge, PostMessageTransport } from "@modelcontextprotocol/ext-apps/app-bridge";
import { HOST_INFO, type ServerView } from "../mcpServer";

export type HostEvent = {
  kind: "tool-input" | "tool-result" | "model-context" | "error";
  text: string;
};

type AppFrameProps = {
  view: ServerView;
  expression: string;
  onEvent: (event: HostEvent) => void;
};

const HOST_CAPABILITIES = { serverTools: {}, updateModelContext: { text: {} }, logging: {} };

export function AppFrame({ view, expression, onEvent }: AppFrameProps) {
  const frame = useRef<HTMLIFrameElement>(null);
  const [height, setHeight] = useState(420);

  useEffect(() => {
    const target = frame.current?.contentWindow;
    if (!target) return;
    const bridge = new AppBridge(view.client, HOST_INFO, HOST_CAPABILITIES);
    bridge.onsizechange = ({ height: next }) => {
      if (next) setHeight(next);
    };
    bridge.onupdatemodelcontext = async ({ content }) => {
      const text = (content ?? []).map((block) => ("text" in block ? block.text : "")).join(" ");
      onEvent({ kind: "model-context", text });
      return {};
    };
    bridge.oninitialized = async () => {
      const args = { expression };
      onEvent({ kind: "tool-input", text: `${view.toolName}(${JSON.stringify(args)})` });
      await bridge.sendToolInput({ arguments: args });
      try {
        const result = await view.client.callTool({ name: view.toolName, arguments: args });
        onEvent({ kind: "tool-result", text: JSON.stringify(result.structuredContent ?? result.content) });
        await bridge.sendToolResult(result);
      } catch (failure) {
        onEvent({ kind: "error", text: failure instanceof Error ? failure.message : String(failure) });
      }
    };
    void bridge.connect(new PostMessageTransport(target, target));
    return () => {
      void bridge.close();
    };
  }, [view, expression, onEvent]);

  return (
    <iframe
      ref={frame}
      className="app-frame"
      title="Calculator MCP App"
      sandbox="allow-scripts"
      srcDoc={view.html}
      style={{ height }}
    />
  );
}
