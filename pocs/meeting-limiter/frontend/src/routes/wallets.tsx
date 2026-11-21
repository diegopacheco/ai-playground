import { createFileRoute } from "@tanstack/react-router";
import { WeekNav } from "../components/WeekNav";
import { WalletTable } from "../features/wallets/components/WalletTable";
import { useWallets } from "../features/wallets/hooks";
import { walletHealth } from "../features/wallets/types";
import { formatMoney } from "../money";
import { Icon } from "../shell/icons";
import { Panel } from "../shell/panels";
import { useWeek } from "../shell/week";

export const Route = createFileRoute("/wallets")({ component: WalletsPage });

function WalletsPage() {
  const week = useWeek();
  const wallets = useWallets(week.weekStart).data ?? [];
  const budget = wallets.reduce((s, w) => s + w.budget, 0);
  const spent = wallets.reduce((s, w) => s + w.spent, 0);
  const broke = wallets.filter((w) => walletHealth(w) === "empty").length;
  return (
    <div className="page page--stack">
      <Panel id="wallet-summary" title="Week summary" icon={<Icon name="coin" />} actions={<WeekNav />}>
        <div className="stats">
          <div className="stat">
            <span>Total budget</span>
            <strong>{formatMoney(budget)}</strong>
          </div>
          <div className="stat">
            <span>Spent on meetings</span>
            <strong>{formatMoney(spent)}</strong>
          </div>
          <div className="stat">
            <span>Left to spend</span>
            <strong>{formatMoney(budget - spent)}</strong>
          </div>
          <div className={`stat ${broke > 0 ? "stat--bad" : ""}`}>
            <span>Empty wallets</span>
            <strong>{broke}</strong>
          </div>
        </div>
      </Panel>
      <Panel id="wallets" title="Wallets" icon={<Icon name="wallet" />} className="grow">
        <WalletTable wallets={wallets} />
      </Panel>
    </div>
  );
}
