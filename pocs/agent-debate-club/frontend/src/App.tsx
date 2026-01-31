import { useState } from 'react';
import { ThemeSetup } from './components/ThemeSetup';
import { DebateView } from './components/DebateView';
import { HistoryList } from './components/HistoryList';

type Screen = 'setup' | 'debate' | 'history';

interface DebateInfo {
  id: string;
  topic: string;
  duration: number;
  agentA: string;
  agentB: string;
  judge: string;
}

function App() {
  const [screen, setScreen] = useState<Screen>('setup');
  const [currentDebate, setCurrentDebate] = useState<DebateInfo | null>(null);

  const handleDebateStarted = (id: string, duration: number, topic: string, agentA: string, agentB: string, judge: string) => {
    setCurrentDebate({ id, topic, duration, agentA, agentB, judge });
    setScreen('debate');
  };

  if (screen === 'history') {
    return <HistoryList onBack={() => setScreen('setup')} />;
  }

  if (screen === 'debate' && currentDebate) {
    return (
      <DebateView
        debateId={currentDebate.id}
        topic={currentDebate.topic}
        duration={currentDebate.duration}
        agentA={currentDebate.agentA}
        agentB={currentDebate.agentB}
        judge={currentDebate.judge}
        onBack={() => {
          setCurrentDebate(null);
          setScreen('setup');
        }}
      />
    );
  }

  return (
    <div className="min-h-screen bg-gray-900">
      <div className="absolute top-4 right-4">
        <button
          onClick={() => setScreen('history')}
          className="px-4 py-2 bg-gray-700 text-white rounded-lg font-medium hover:bg-gray-600 transition border border-gray-600"
        >
          View History
        </button>
      </div>
      <ThemeSetup onDebateStarted={handleDebateStarted} />
    </div>
  );
}

export default App;
