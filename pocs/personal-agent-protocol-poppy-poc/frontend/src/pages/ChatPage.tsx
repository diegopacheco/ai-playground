import { useForm } from "@tanstack/react-form";
import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import { conversationEndpoint, type Agent } from "../agent";
import { Card, ErrorLine, WithAgent } from "../components";
import { randomId } from "../poppy/b64";
import type { ChatMessage, ConversationSummary, EventsPage } from "../poppy/types";

function Chat({ agent }: { agent: Agent }) {
  const endpoint = conversationEndpoint(agent.document);
  const [summary, setSummary] = useState<ConversationSummary | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [cursor, setCursor] = useState("");
  const send = useMutation({
    mutationFn: async (text: string) => {
      const message = { id: randomId("msg_"), author: { type: "agent" }, text };
      const current = summary
        ? await agent.client.call<ConversationSummary>(`${endpoint}/${summary.conversation_id}/messages`, { method: "POST", body: { message } })
        : await agent.client.call<ConversationSummary>(endpoint, { method: "POST", body: { message: { ...message, context: { locale: "en-US", user_available: true } } } });
      const page = await agent.client.call<EventsPage>(`${endpoint}/${current.conversation_id}/events${summary ? `?cursor=${cursor}` : ""}`);
      return page;
    },
    onSuccess: (page) => {
      setSummary(page);
      setCursor(page.cursor);
      setMessages((previous) => [...previous, ...page.events.flatMap((event) => (event.message ? [event.message] : []))]);
    },
  });
  const close = useMutation({
    mutationFn: (id: string) => agent.client.call<ConversationSummary>(`${endpoint}/${id}/close`, { method: "POST" }),
    onSuccess: setSummary,
  });
  const form = useForm({
    defaultValues: { text: "" },
    onSubmit: ({ value, formApi }) => {
      if (!value.text.trim()) return;
      send.mutate(value.text);
      formApi.reset();
    },
  });
  const restart = () => {
    setSummary(null);
    setMessages([]);
    setCursor("");
  };
  return (
    <Card title={`Conversation with the Willow Inn Company Agent ${summary ? `(${summary.conversation_id}, ${summary.status}, responder ${summary.responder})` : ""}`}>
      <div className="chat">
        {messages.length === 0 ? <p className="dim">Try: "Do you allow dogs?", "When is check-in?", "Can I pay with a card?"</p> : null}
        {messages.map((message) => (
          <div key={message.id} className={`bubble ${message.role}`}>
            <span className="who">{message.role === "user" ? "Atlas (for the user)" : `Willow Inn ${message.author.type}`}</span>
            {message.text}
          </div>
        ))}
      </div>
      <form
        className="row"
        onSubmit={(event) => {
          event.preventDefault();
          form.handleSubmit();
        }}
      >
        <form.Field name="text">{(field) => <input className="grow" placeholder="Message the Company Agent" value={field.state.value} onChange={(event) => field.handleChange(event.target.value)} />}</form.Field>
        <button type="submit" disabled={send.isPending || summary?.status === "closed"}>
          Send
        </button>
        {summary && summary.status !== "closed" ? <button type="button" onClick={() => close.mutate(summary.conversation_id)}>Close</button> : null}
        {summary?.status === "closed" ? <button type="button" onClick={restart}>New conversation</button> : null}
      </form>
      <ErrorLine error={send.error ?? close.error} />
    </Card>
  );
}

export function ChatPage() {
  return <WithAgent>{(agent) => <Chat agent={agent} />}</WithAgent>;
}
