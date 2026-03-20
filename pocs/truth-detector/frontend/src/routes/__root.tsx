import { createRootRoute, Link, Outlet } from "@tanstack/react-router";

export const Route = createRootRoute({
  component: RootLayout,
});

function RootLayout() {
  return (
    <div className="bg-gray-950 text-gray-100 min-h-screen">
      <header className="border-b border-gray-800 px-6 py-4">
        <h1 className="text-2xl font-bold text-white mb-4">Truth Detector</h1>
        <nav className="flex gap-6">
          <Link
            to="/"
            className="pb-2 text-sm font-medium transition-colors hover:text-white"
            activeProps={{
              className:
                "pb-2 text-sm font-medium text-white border-b-2 border-blue-500",
            }}
            inactiveProps={{
              className: "pb-2 text-sm font-medium text-gray-400 hover:text-white",
            }}
          >
            Analyze User
          </Link>
          <Link
            to="/leaderboard"
            className="pb-2 text-sm font-medium transition-colors hover:text-white"
            activeProps={{
              className:
                "pb-2 text-sm font-medium text-white border-b-2 border-blue-500",
            }}
            inactiveProps={{
              className: "pb-2 text-sm font-medium text-gray-400 hover:text-white",
            }}
          >
            Leaderboard
          </Link>
        </nav>
      </header>
      <main className="p-6 max-w-5xl mx-auto">
        <Outlet />
      </main>
    </div>
  );
}
