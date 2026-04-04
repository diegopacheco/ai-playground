import { useState, useMemo } from 'react';
import type { Airport, SearchRequest } from '../types';
import { airports } from '../data/airports';

interface Props {
  onSearch: (req: SearchRequest) => void;
  loading: boolean;
}

function AirportInput({ label, value, onChange }: {
  label: string;
  value: Airport | null;
  onChange: (a: Airport) => void;
}) {
  const [query, setQuery] = useState('');
  const [open, setOpen] = useState(false);

  const filtered = useMemo(() => {
    if (!query) return [];
    const q = query.toLowerCase();
    return airports.filter(a =>
      a.code.toLowerCase().includes(q) ||
      a.city.toLowerCase().includes(q) ||
      a.country.toLowerCase().includes(q) ||
      a.name.toLowerCase().includes(q)
    ).slice(0, 8);
  }, [query]);

  return (
    <div className="relative">
      <label className="block text-sm font-medium text-gray-300 mb-1">{label}</label>
      <input
        type="text"
        className="w-full px-4 py-3 bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        placeholder="Type city, country, or airport code..."
        value={value ? `${value.code} - ${value.city}, ${value.country}` : query}
        onChange={e => {
          setQuery(e.target.value);
          setOpen(true);
          if (value) onChange(null as unknown as Airport);
        }}
        onFocus={() => { if (query) setOpen(true); }}
        onBlur={() => setTimeout(() => setOpen(false), 200)}
      />
      {open && filtered.length > 0 && (
        <ul className="absolute z-50 w-full mt-1 bg-gray-800 border border-gray-600 rounded-lg shadow-xl max-h-60 overflow-auto">
          {filtered.map(a => (
            <li
              key={a.code}
              className="px-4 py-3 hover:bg-gray-700 cursor-pointer text-white border-b border-gray-700 last:border-0"
              onMouseDown={() => {
                onChange(a);
                setQuery('');
                setOpen(false);
              }}
            >
              <span className="font-bold text-blue-400">{a.code}</span>
              <span className="ml-2">{a.city}, {a.country}</span>
              <span className="ml-2 text-gray-500 text-sm">({a.name})</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function SearchForm({ onSearch, loading }: Props) {
  const [origin, setOrigin] = useState<Airport | null>(null);
  const [destination, setDestination] = useState<Airport | null>(null);
  const [date, setDate] = useState(() => {
    const d = new Date();
    d.setDate(d.getDate() + 14);
    return d.toISOString().split('T')[0];
  });
  const [agent, setAgent] = useState('claude');

  const canSearch = origin && destination && date && !loading;

  const handleSubmit = () => {
    if (!origin || !destination) return;
    onSearch({
      origin: origin.code,
      origin_city: `${origin.city}, ${origin.country}`,
      destination: destination.code,
      destination_city: `${destination.city}, ${destination.country}`,
      date,
      agent,
    });
  };

  return (
    <div className="bg-gray-850 border border-gray-700 rounded-2xl p-8 shadow-2xl">
      <div className="flex items-center gap-3 mb-8">
        <div className="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center text-white text-xl font-bold">F</div>
        <h1 className="text-2xl font-bold text-white">Flight Agent MCP</h1>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        <AirportInput label="From" value={origin} onChange={setOrigin} />
        <AirportInput label="To" value={destination} onChange={setDestination} />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">Date</label>
          <input
            type="date"
            className="w-full px-4 py-3 bg-gray-800 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            value={date}
            onChange={e => setDate(e.target.value)}
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">Agent</label>
          <select
            className="w-full px-4 py-3 bg-gray-800 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            value={agent}
            onChange={e => setAgent(e.target.value)}
          >
            <option value="claude">Claude</option>
            <option value="codex">Codex</option>
          </select>
        </div>
      </div>

      <button
        onClick={handleSubmit}
        disabled={!canSearch}
        className={`w-full py-4 rounded-lg font-semibold text-lg transition-all ${
          canSearch
            ? 'bg-blue-600 hover:bg-blue-500 text-white shadow-lg hover:shadow-xl'
            : 'bg-gray-700 text-gray-500 cursor-not-allowed'
        }`}
      >
        {loading ? (
          <span className="flex items-center justify-center gap-3">
            <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Agent is searching...
          </span>
        ) : (
          'Search Flights'
        )}
      </button>
    </div>
  );
}
