import { catalog } from "../domain/catalog";
import type { PrincipleGroup } from "../domain/types";

const GROUPS: readonly PrincipleGroup[] = ["Core", "Architecture", "Verification", "Delegation", "Meta"];

export function PrinciplesPage() {
  return (
    <div>
      {GROUPS.map((g) => (
        <section key={g}>
          <h2>{g}</h2>
          <div className="grid">
            {catalog.principles
              .filter((p) => p.group === g)
              .map((p) => (
                <article className="card" key={p.name}>
                  <h3>{p.name}</h3>
                  <p>{p.when}</p>
                </article>
              ))}
          </div>
        </section>
      ))}
    </div>
  );
}
