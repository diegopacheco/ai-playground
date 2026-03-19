import { createFileRoute } from "@tanstack/react-router";
import { HistoryTable } from "../components/HistoryTable.tsx";
import { useAuctions } from "../hooks/useAuctions.ts";

export const Route = createFileRoute("/history")({
  component: HistoryPage,
});

function HistoryPage() {
  const { data: auctions, isLoading } = useAuctions();

  return (
    <div className="max-w-6xl mx-auto">
      <h2 className="text-2xl font-bold text-amber-400 mb-6">
        Auction History
      </h2>
      {isLoading ? (
        <div className="text-gray-400 text-center py-8">Loading...</div>
      ) : (
        <HistoryTable auctions={auctions || []} />
      )}
    </div>
  );
}
