import { useEffect, useState, useCallback } from 'react';
import { getDebateStreamUrl } from '../api/debates';
import type { Message, DebateResult, DebateStatus, DebateEvent } from '../types';

export function useDebateSSE(debateId: string | null) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [status, setStatus] = useState<DebateStatus>('idle');
  const [thinkingAgent, setThinkingAgent] = useState<string | null>(null);
  const [result, setResult] = useState<DebateResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const reset = useCallback(() => {
    setMessages([]);
    setStatus('idle');
    setThinkingAgent(null);
    setResult(null);
    setError(null);
  }, []);

  useEffect(() => {
    if (!debateId) return;

    reset();
    const es = new EventSource(getDebateStreamUrl(debateId));

    es.addEventListener('agent_thinking', (e) => {
      const data: DebateEvent = JSON.parse(e.data);
      if (data.type === 'agent_thinking') {
        setStatus('thinking');
        setThinkingAgent(data.agent);
      }
    });

    es.addEventListener('agent_message', (e) => {
      const data: DebateEvent = JSON.parse(e.data);
      if (data.type === 'agent_message') {
        setStatus('idle');
        setThinkingAgent(null);
        setMessages((prev) => [
          ...prev,
          {
            id: prev.length + 1,
            debate_id: debateId,
            agent: data.agent,
            content: data.content,
            stance: data.stance,
            created_at: new Date().toISOString(),
          },
        ]);
      }
    });

    es.addEventListener('debate_over', (e) => {
      const data: DebateEvent = JSON.parse(e.data);
      if (data.type === 'debate_over') {
        setStatus('over');
        setThinkingAgent(null);
        setResult({
          winner: data.winner,
          reason: data.reason,
          duration_ms: data.duration_ms,
        });
        es.close();
      }
    });

    es.addEventListener('error', (e) => {
      if (e instanceof MessageEvent) {
        const data: DebateEvent = JSON.parse(e.data);
        if (data.type === 'error') {
          setError(data.message);
        }
      }
    });

    es.onerror = () => {
      es.close();
    };

    return () => {
      es.close();
    };
  }, [debateId, reset]);

  return { messages, status, thinkingAgent, result, error };
}
