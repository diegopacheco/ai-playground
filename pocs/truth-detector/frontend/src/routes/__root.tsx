import { createRootRoute, Link, Outlet } from "@tanstack/react-router";

export const Route = createRootRoute({
  component: RootLayout,
});

function RootLayout() {
  return (
    <div className="bg-white text-gray-900 min-h-screen">
      <header className="border-b border-gray-200 px-6 py-4">
        <h1 className="text-2xl font-bold text-gray-900 mb-4">Truth Detector</h1>
        <nav className="flex gap-6">
          <Link
            to="/"
            className="pb-2 text-sm font-medium transition-colors hover:text-gray-900"
            activeProps={{
              className:
                "pb-2 text-sm font-medium text-gray-900 border-b-2 border-blue-500",
            }}
            inactiveProps={{
              className: "pb-2 text-sm font-medium text-gray-500 hover:text-gray-900",
            }}
          >
            Analyze User
          </Link>
          <Link
            to="/leaderboard"
            className="pb-2 text-sm font-medium transition-colors hover:text-gray-900"
            activeProps={{
              className:
                "pb-2 text-sm font-medium text-gray-900 border-b-2 border-blue-500",
            }}
            inactiveProps={{
              className: "pb-2 text-sm font-medium text-gray-500 hover:text-gray-900",
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
