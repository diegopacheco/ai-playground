import { useState } from 'react';
import { SearchForm } from './components/SearchForm';
import { ResultsGrid } from './components/ResultsGrid';
import { searchFlights } from './api/flights';
import type { SearchRequest, FlightResult } from './types';

function App() {
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<FlightResult[]>([]);
  const [agentUsed, setAgentUsed] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [rawOutput, setRawOutput] = useState('');
  const [searched, setSearched] = useState(false);

  const handleSearch = async (req: SearchRequest) => {
    setLoading(true);
    setError(null);
    setResults([]);
    setRawOutput('');
    setSearched(true);
    try {
      const response = await searchFlights(req);
      setResults(response.results);
      setAgentUsed(response.agent_used);
      setError(response.error);
      setRawOutput(response.raw_output || '');
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to connect to backend');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 py-8 px-4">
      <div className="max-w-6xl mx-auto">
        <SearchForm onSearch={handleSearch} loading={loading} />
        {searched && !loading && (
          <ResultsGrid results={results} agentUsed={agentUsed} error={error} rawOutput={rawOutput} />
        )}
        {loading && (
          <div className="mt-8 text-center">
            <div className="inline-flex items-center gap-3 bg-gray-800 border border-gray-700 rounded-xl px-6 py-4">
              <svg className="animate-spin h-6 w-6 text-blue-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              <span className="text-gray-300 text-lg">Agent is searching for flights... This may take a minute.</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
