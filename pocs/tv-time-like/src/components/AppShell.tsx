import { Link, Outlet, useRouterState } from "@tanstack/react-router"
import { Brand } from "./Brand"
import { Icon } from "./Icon"

const items = [
  { to: "/", label: "Discover", icon: "search" as const },
  { to: "/library", label: "My library", icon: "library" as const },
  { to: "/metrics", label: "Metrics", icon: "metrics" as const }
]

export function AppShell() {
  const pathname = useRouterState({ select: state => state.location.pathname })
  return <div className="app-shell">
    <header className="topbar">
      <Link to="/" className="brand-link"><Brand/></Link>
      <nav aria-label="Main navigation">
        {items.map(item => <Link key={item.to} to={item.to} className={pathname === item.to ? "nav-item active" : "nav-item"}>
          <Icon name={item.icon} size={18}/><span>{item.label}</span>
        </Link>)}
      </nav>
      <div className="profile"><span>DP</span><div><strong>Your journal</strong><small>Private collection</small></div></div>
    </header>
    <main><Outlet/></main>
    <footer><Brand/><span>Your viewing history, kept local.</span><span>TV data by TVmaze · Movie data by TMDB</span></footer>
  </div>
}
