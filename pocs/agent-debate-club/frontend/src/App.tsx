import { useState } from 'react';
import { ThemeSetup } from './components/ThemeSetup';
import { DebateView } from './components/DebateView';
import { HistoryList } from './components/HistoryList';

type Screen = 'setup' | 'debate' | 'history';

interface DebateInfo {
  id: string;
  topic: string;
  duration: number;
}

function App() {
  const [screen, setScreen] = useState<Screen>('setup');
  const [currentDebate, setCurrentDebate] = useState<DebateInfo | null>(null);

  const handleDebateStarted = (id: string, duration: number, topic: string) => {
    setCurrentDebate({ id, topic, duration });
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
          className="text-gray-400 hover:text-white transition"
        >
          History
        </button>
      </div>
      <ThemeSetup onDebateStarted={handleDebateStarted} />
    </div>
  );
}

export default App;
