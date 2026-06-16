import { Link, Outlet } from "@tanstack/react-router";

export function RootLayout() {
  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">Pixel Pantry</div>
        <nav className="nav">
          <Link to="/">Shop</Link>
          <Link to="/about">About</Link>
        </nav>
        <button className="icon-btn cart" />
      </header>

      <div className="content">
        <Outlet />
      </div>

      <footer className="foot">
        <span className="muted">
          Free shipping over $40. <a href="/about">Read more</a>
        </span>
        <span className="faint">© 2026 Pixel Pantry</span>
      </footer>
    </div>
  );
}
