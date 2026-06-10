import React from "react";
import _ from "lodash";
import { format, parseISO } from "date-fns";

export default function StatsPanel({ releases }) {
  const success = _.filter(releases, { status: "success" }).length;
  const rate = Math.round((success / releases.length) * 100);
  const avg = Math.round(_.meanBy(releases, "durationMs") / 1000);
  const last = _.maxBy(releases, (r) => parseISO(r.at).getTime());

  return (
    <section className="stats">
      <div className="stat">
        <div className="k">Success rate</div>
        <div className="v">{rate}%</div>
      </div>
      <div className="stat">
        <div className="k">Avg duration</div>
        <div className="v">{avg}s</div>
      </div>
      <div className="stat">
        <div className="k">Last deploy</div>
        <div className="v">{format(parseISO(last.at), "MMM d, HH:mm")}</div>
        <div className="n">{last.service}</div>
      </div>
    </section>
  );
}
