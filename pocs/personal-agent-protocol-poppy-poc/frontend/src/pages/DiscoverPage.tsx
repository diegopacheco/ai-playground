import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import type { Agent } from "../agent";
import { Card, JsonBlock, WithAgent } from "../components";
import { newUserFor, userIdFor } from "../poppy/identity";
import { decodeJwt } from "../poppy/jwt";
import type { ExcludedTasks } from "../poppy/types";

function Discover({ agent }: { agent: Agent }) {
  const { client, document } = agent;
  const queryClient = useQueryClient();
  const [userId, setUserId] = useState(() => userIdFor(client.identity, client.audience));
  const excluded = useQuery({
    queryKey: ["excluded"],
    queryFn: () => client.call<ExcludedTasks>(document.excluded_tasks?.url ?? "", { session: false }),
    enabled: Boolean(document.excluded_tasks),
  });
  const switchUser = () => {
    setUserId(newUserFor(client.identity, client.audience));
    queryClient.invalidateQueries({ queryKey: ["holds"] });
  };
  return (
    <div className="grid">
      <Card title={`Discovered ${document.organization.name}`}>
        <p className="dim">GET /.well-known/poppy.json</p>
        <JsonBlock value={document} />
      </Card>
      <div className="stack">
        <Card title="Personal Agent identity">
          <dl>
            <dt>client_id</dt>
            <dd>{client.identity.clientId}</dd>
            <dt>root key kid</dt>
            <dd>{client.identity.rootJwk.kid}</dd>
            <dt>delegated key kid</dt>
            <dd>{client.identity.delegatedKid}</dd>
            <dt>pairwise User ID at this Company</dt>
            <dd>
              <code>{userId}</code> <button onClick={switchUser}>Act for another user</button>
            </dd>
          </dl>
          <p className="dim">The Company only ever sees the pairwise User ID. It contains no personal data and differs per Company.</p>
          <h3>Delegation JWT</h3>
          <JsonBlock value={decodeJwt(client.identity.delegation)} />
        </Card>
        <Card title="Tasks Willow Inn does not offer through Poppy">
          <ul>
            {excluded.data?.tasks.map((task) => (
              <li key={task.task}>
                <strong>{task.task}</strong>
                {task.alternative ? <span className="dim"> - {task.alternative.instructions}</span> : null}
              </li>
            ))}
          </ul>
        </Card>
      </div>
    </div>
  );
}

export function DiscoverPage() {
  return <WithAgent>{(agent) => <Discover agent={agent} />}</WithAgent>;
}
