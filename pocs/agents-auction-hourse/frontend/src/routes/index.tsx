import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useState } from "react";
import { AuctionSetup } from "../components/AuctionSetup.tsx";
import { AuctionLive } from "../components/AuctionLive.tsx";
import { useCreateAuction } from "../hooks/useAuctions.ts";
import type { Agent } from "../types/index.ts";

export const Route = createFileRoute("/")({
  component: HomePage,
});

function HomePage() {
  const navigate = useNavigate();
  const createAuction = useCreateAuction();
  const [auctionId, setAuctionId] = useState<string | null>(null);
  const [agents, setAgents] = useState<Agent[]>([]);

  async function handleStart(selectedAgents: Agent[]) {
    setAgents(selectedAgents);
    const auction = await createAuction.mutateAsync(selectedAgents);
    setAuctionId(auction.id);
  }

  if (auctionId) {
    return <AuctionLive auctionId={auctionId} agents={agents} />;
  }

  return (
    <AuctionSetup onStart={handleStart} isLoading={createAuction.isPending} />
  );
}
