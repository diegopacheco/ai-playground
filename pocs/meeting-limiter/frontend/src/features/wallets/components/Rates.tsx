import { RoleBadge } from "../../../components/Person";
import { formatMoney } from "../../../money";
import { usePolicies } from "../../people/hooks";

export function Rates() {
  const { data } = usePolicies();
  return (
    <table className="data-table compact">
      <thead>
        <tr>
          <th>Role</th>
          <th>Hours/week</th>
          <th>Rate</th>
          <th>Wallet</th>
        </tr>
      </thead>
      <tbody>
        {(data ?? []).map((p) => (
          <tr key={p.role}>
            <td>
              <RoleBadge role={p.role} />
            </td>
            <td>{p.weeklyHours}h</td>
            <td>{formatMoney(p.hourlyRate)}/h</td>
            <td>{formatMoney(p.weeklyBudget)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
