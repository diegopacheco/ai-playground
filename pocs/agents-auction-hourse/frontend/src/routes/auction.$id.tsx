import { createFileRoute } from "@tanstack/react-router";
import { AuctionLive } from "../components/AuctionLive.tsx";
import { useAuction } from "../hooks/useAuctions.ts";

export const Route = createFileRoute("/auction/$id")({
  component: AuctionPage,
});

function AuctionPage() {
  const { id } = Route.useParams();
  const { data: auction } = useAuction(id);

  const agents = (auction?.agents || []).map((a) => ({
    name: a.agent_name,
    model: a.model,
    budget: a.initial_budget,
  }));

  return (
    <div>
      <AuctionLive auctionId={id} agents={agents} />
    </div>
  );
}
