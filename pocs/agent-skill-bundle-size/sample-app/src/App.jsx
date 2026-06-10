import React, { useMemo } from "react";
import _ from "lodash";
import { releases } from "./data.js";
import StatsPanel from "./components/StatsPanel.jsx";
import Notes from "./components/Notes.jsx";

export default function App() {
  const byService = useMemo(() => _.groupBy(releases, "service"), []);
  const services = Object.keys(byService).sort();

  return (
    <div className="page">
      <header>
        <h1>Releases Dashboard</h1>
        <p>{releases.length} deploys across {services.length} services</p>
      </header>

      <StatsPanel releases={releases} />

      <section className="services">
        {services.map((name) => {
          const group = byService[name];
          const failed = _.filter(group, (r) => r.status !== "success").length;
          return (
            <div key={name} className="svc">
              <h3>{name}</h3>
              <div className="count">{group.length} deploys</div>
              <div className={failed ? "warn" : "ok"}>
                {failed ? `${failed} need attention` : "all healthy"}
              </div>
            </div>
          );
        })}
      </section>

      <Notes />
    </div>
  );
}
