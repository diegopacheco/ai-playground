import { createRootRoute, Link, Outlet } from "@tanstack/react-router";

export const Route = createRootRoute({
  component: RootLayout,
});

function RootLayout() {
  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <nav className="bg-gray-800 border-b border-gray-700 px-6 py-3">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <span className="text-amber-400 font-bold text-xl">
            Agent Auction House
          </span>
          <div className="flex gap-4">
            <Link
              to="/"
              className="text-gray-300 hover:text-amber-400 transition-colors [&.active]:text-amber-400"
            >
              Home
            </Link>
            <Link
              to="/history"
              className="text-gray-300 hover:text-amber-400 transition-colors [&.active]:text-amber-400"
            >
              History
            </Link>
          </div>
        </div>
      </nav>
      <main className="p-6">
        <Outlet />
      </main>
    </div>
  );
}
