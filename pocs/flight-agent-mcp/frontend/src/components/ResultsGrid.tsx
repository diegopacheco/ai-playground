import type { FlightResult } from '../types';

interface Props {
  results: FlightResult[];
  agentUsed: string;
  error: string | null;
  rawOutput: string;
}

function sourceColor(source: string): string {
  if (source.includes('seats')) return 'bg-purple-600';
  if (source.includes('google')) return 'bg-green-600';
  return 'bg-gray-600';
}

function cabinColor(cabin: string): string {
  const c = cabin.toLowerCase();
  if (c.includes('first')) return 'text-yellow-400';
  if (c.includes('business')) return 'text-blue-400';
  if (c.includes('premium')) return 'text-teal-400';
  return 'text-gray-300';
}

export function ResultsGrid({ results, agentUsed, error, rawOutput }: Props) {
  if (error) {
    return (
      <div className="bg-red-900/30 border border-red-700 rounded-xl p-6 mt-6">
        <h3 className="text-red-400 font-semibold text-lg mb-2">Search Failed</h3>
        <p className="text-red-300 mb-4">{error}</p>
        {rawOutput && (
          <details className="mt-2">
            <summary className="text-gray-400 cursor-pointer text-sm hover:text-gray-300">Show raw agent output</summary>
            <pre className="mt-2 p-4 bg-gray-900 rounded-lg text-xs text-gray-300 overflow-x-auto whitespace-pre-wrap max-h-96 overflow-y-auto">{rawOutput}</pre>
          </details>
        )}
      </div>
    );
  }

  if (results.length === 0) {
    return (
      <div className="bg-gray-800 border border-gray-700 rounded-xl p-8 mt-6 text-center">
        <p className="text-gray-400 text-lg">No flights found. Try different airports or dates.</p>
      </div>
    );
  }

  return (
    <div className="mt-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold text-white">
          {results.length} Flight{results.length !== 1 ? 's' : ''} Found
        </h2>
        <span className="text-sm text-gray-400">
          Searched by: <span className="text-blue-400 font-medium">{agentUsed}</span>
        </span>
      </div>

      <div className="bg-gray-800 border border-gray-700 rounded-xl overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="bg-gray-750 border-b border-gray-700">
                <th className="px-4 py-3 text-left text-xs font-semibold text-gray-400 uppercase tracking-wider">Date</th>
                <th className="px-4 py-3 text-left text-xs font-semibold text-gray-400 uppercase tracking-wider">Route</th>
                <th className="px-4 py-3 text-left text-xs font-semibold text-gray-400 uppercase tracking-wider">Airline</th>
                <th className="px-4 py-3 text-left text-xs font-semibold text-gray-400 uppercase tracking-wider">Price</th>
                <th className="px-4 py-3 text-left text-xs font-semibold text-gray-400 uppercase tracking-wider">Cabin</th>
                <th className="px-4 py-3 text-left text-xs font-semibold text-gray-400 uppercase tracking-wider">Stops</th>
                <th className="px-4 py-3 text-left text-xs font-semibold text-gray-400 uppercase tracking-wider">Duration</th>
                <th className="px-4 py-3 text-left text-xs font-semibold text-gray-400 uppercase tracking-wider">Source</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-700">
              {results.map((flight) => {
                const hasUrl = flight.booking_url && flight.booking_url.length > 0;
                return (
                  <tr
                    key={flight.id}
                    className={`transition-colors ${hasUrl ? 'hover:bg-gray-700 cursor-pointer' : 'hover:bg-gray-750'}`}
                    onClick={() => {
                      if (hasUrl) window.open(flight.booking_url, '_blank');
                    }}
                    title={hasUrl ? 'Click to book' : ''}
                  >
                    <td className="px-4 py-4 text-sm text-white whitespace-nowrap">{flight.date}</td>
                    <td className="px-4 py-4 text-sm text-white whitespace-nowrap font-medium">
                      {flight.origin} → {flight.destination}
                    </td>
                    <td className="px-4 py-4 text-sm text-gray-300 whitespace-nowrap">{flight.airline}</td>
                    <td className="px-4 py-4 text-sm text-emerald-400 whitespace-nowrap font-semibold">{flight.price}</td>
                    <td className={`px-4 py-4 text-sm whitespace-nowrap font-medium ${cabinColor(flight.cabin)}`}>
                      {flight.cabin}
                    </td>
                    <td className="px-4 py-4 text-sm text-gray-300 whitespace-nowrap">{flight.stops}</td>
                    <td className="px-4 py-4 text-sm text-gray-300 whitespace-nowrap">{flight.duration}</td>
                    <td className="px-4 py-4 text-sm whitespace-nowrap">
                      <span className={`px-2 py-1 rounded-full text-xs font-medium text-white ${sourceColor(flight.source)}`}>
                        {flight.source}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      <p className="text-gray-500 text-xs mt-3 text-center">
        Click on a row with a booking link to open it in a new tab
      </p>
    </div>
  );
}
