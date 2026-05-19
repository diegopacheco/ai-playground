// SideProjectStack.jsx — 🥇 Tiny Side Projects. Auto-rotates every 3s.
// Pause on hover; click opens the GitHub repo in a new tab.

const PROJECTS = [
  { emoji: '🧝🏾‍♂️', name: 'Tupi lang', desc: 'Java 23 PL',         slug: 'tupilang',   href: 'https://github.com/diegopacheco/tupilang' },
  { emoji: '🥫',     name: 'Jello',     desc: 'Vanilla JS Trello',    slug: 'jello',      href: 'https://github.com/diegopacheco/jello' },
  { emoji: '📑',     name: 'Zim',       desc: 'Zig 0.13 vim-like',    slug: 'zim',        href: 'https://github.com/diegopacheco/zim' },
  { emoji: '💻',     name: 'Gorminator',desc: 'Go Linux terminal',    slug: 'gorminator', href: 'https://github.com/diegopacheco/gorminator' },
  { emoji: '😸',     name: 'kit',       desc: 'Git-like in Kotlin',   slug: 'kit',        href: 'https://github.com/diegopacheco/kit' },
  { emoji: '🦀',     name: 'Shrust',    desc: 'Rust compression',     slug: 'shrust',     href: 'https://github.com/diegopacheco/shrust' },
  { emoji: '🕵🏽',     name: 'Smith',     desc: 'Scala 3 security agent', slug: 'smith',   href: 'https://github.com/diegopacheco/smith' },
  { emoji: '📟',     name: 'ZOS',       desc: 'Tiny Zig OS',          slug: 'zos',        href: 'https://github.com/diegopacheco/zos' },
  { emoji: '🎮',     name: 'Tiny Games',desc: 'Collection of JS games', slug: 'Tiny Games', href: 'https://gist.github.com/diegopacheco/d48104e8f584e3209ce7d5f5c0186e0e' },
];

function SideProjectCard({ p, index, isActive }) {
  return (
    <a
      className={'side-project-card' + (isActive ? ' is-active' : '')}
      href={p.href}
      title={`${p.name} on GitHub`}
      target="_blank"
      rel="noopener noreferrer"
      style={{ '--stack-index': index }}
    >
      <div className="side-project-card-header">
        <span className="side-project-emoji" aria-hidden="true">{p.emoji}</span>
        <div>
          <p className="side-project-name">{p.name}</p>
          <p className="side-project-desc">{p.desc}</p>
        </div>
      </div>
      <span className="side-project-link">{p.slug}</span>
    </a>
  );
}

function SideProjectStack() {
  const [order, setOrder] = React.useState(PROJECTS);
  const [paused, setPaused] = React.useState(false);

  React.useEffect(() => {
    if (paused) return;
    const t = setInterval(() => {
      setOrder((prev) => [...prev.slice(1), prev[0]]);
    }, 3000);
    return () => clearInterval(t);
  }, [paused]);

  return (
    <aside className="side-projects-stack">
      <h3>🥇 Tiny Side Projects</h3>
      <div
        className="side-projects-list"
        onMouseEnter={() => setPaused(true)}
        onMouseLeave={() => setPaused(false)}
      >
        {order.map((p, i) => (
          <SideProjectCard
            key={p.slug}
            p={p}
            index={i}
            isActive={i === 0}
          />
        ))}
      </div>
    </aside>
  );
}

window.SideProjectStack = SideProjectStack;
