import { Meter } from "../../../components/Meter";
import { PersonCell } from "../../../components/Person";
import { formatMinutes, formatMoney } from "../../../money";
import { walletHealth, type Wallet } from "../types";

export function MyWallet({ wallet }: { readonly wallet: Wallet | undefined }) {
  if (!wallet) return <p className="muted">Loading wallet...</p>;
  const health = walletHealth(wallet);
  return (
    <div className="my-wallet">
      <PersonCell user={wallet.user} />
      <div className={`balance balance--${health}`}>{formatMoney(wallet.balance)}</div>
      <Meter value={wallet.spent} max={wallet.budget} tone={health} label="budget spent" />
      <dl>
        <dt>Spent</dt>
        <dd>
          {formatMoney(wallet.spent)} of {formatMoney(wallet.budget)}
        </dd>
        <dt>Meeting time</dt>
        <dd>
          {formatMinutes(wallet.minutesUsed)} of {formatMinutes(wallet.minutesLimit)}
        </dd>
        <dt>Meetings</dt>
        <dd>{wallet.meetings}</dd>
      </dl>
      {health === "empty" && <p className="quote-warning">Wallet is empty. New meetings unlock when it refills next Monday.</p>}
    </div>
  );
}
