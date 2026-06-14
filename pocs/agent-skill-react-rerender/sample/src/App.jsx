import { useEffect, useState } from "react";
import StatBadge from "./components/StatBadge.jsx";
import ProductCard from "./components/ProductCard.jsx";
import SearchBar from "./components/SearchBar.jsx";
import ActivityFeed from "./components/ActivityFeed.jsx";
import MetricsTable from "./components/MetricsTable.jsx";
import { products, kpis } from "./data.js";

export default function App() {
  const [term, setTerm] = useState("");
  const [clock, setClock] = useState(() => new Date().toLocaleTimeString());

  useEffect(() => {
    const id = setInterval(() => setClock(new Date().toLocaleTimeString()), 1000);
    return () => clearInterval(id);
  }, []);

  const shown = term
    ? products.filter((p) => p.name.toLowerCase().includes(term.toLowerCase()))
    : products;

  return (
    <div className="app">
      <header className="app__bar">
        <div className="app__brand">
          <span className="app__logo">◆</span>
          <span>Nimbus Commerce Console</span>
        </div>
        <SearchBar onSearch={setTerm} />
        <span className="app__clock">{clock}</span>
      </header>

      <section className="app__kpis">
        <StatBadge label="Revenue" value={Math.round(kpis.revenue)} unit="$" delta={8.4} />
        <StatBadge label="Units" value={kpis.units} delta={3.1} />
        <StatBadge label="Orders" value={kpis.orders} delta={-2.2} />
        <StatBadge
          label="Refund rate"
          value={Math.round(kpis.refundRate * 1000) / 10}
          unit="%"
          delta={-0.6}
        />
      </section>

      <section className="app__grid">
        {shown.map((p) => (
          <ProductCard key={p.id} product={p} />
        ))}
      </section>

      <div className="app__cols">
        <ActivityFeed />
        <MetricsTable />
      </div>
    </div>
  );
}
