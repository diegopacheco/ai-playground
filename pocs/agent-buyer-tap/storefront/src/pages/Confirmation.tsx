import { useParams } from '@tanstack/react-router';
import { useQuery } from '@tanstack/react-query';
import { getOrder, money } from '../api';

export function Confirmation() {
  const { id } = useParams({ from: '/orders/$id' });
  const { data, isLoading, isError } = useQuery({ queryKey: ['order', id], queryFn: () => getOrder(id) });

  if (isLoading) return <p className="muted">loading order...</p>;
  if (isError || !data) return <p className="muted">order not found</p>;

  return (
    <section className="confirm">
      <span className="badge">{data.status}</span>
      <h1>Order confirmed</h1>
      <p className="muted">Paid by an AI agent, verified by signature and consent.</p>
      <ul className="items">
        {data.items.map((it, i) => <li key={i}><span>{it.qty} x</span> {it.sku}</li>)}
      </ul>
      <div className="total">{money(data.amountCents, data.currency)}</div>
      <dl className="meta">
        <dt>order id</dt><dd className="mono">{data.id}</dd>
        <dt>buyer</dt><dd>{data.buyer}</dd>
        <dt>agent key</dt><dd className="mono">{data.agentKeyid}</dd>
        <dt>wallet key</dt><dd className="mono">{data.walletKid}</dd>
        <dt>placed</dt><dd>{data.createdAt}</dd>
      </dl>
    </section>
  );
}
