import { useAuth } from "../hooks/useAuth";

interface NavbarProps {
  onNavigate: (page: string, params?: Record<string, string>) => void;
}

export function Navbar({ onNavigate }: NavbarProps) {
  const { user, logout, isAuthenticated } = useAuth();

  const handleLogout = () => {
    logout();
    onNavigate("login");
  };

  return (
    <nav className="bg-blue-600 text-white shadow-lg">
      <div className="max-w-4xl mx-auto px-4 py-3 flex items-center justify-between">
        <button
          onClick={() => onNavigate("home")}
          className="text-xl font-bold tracking-tight"
        >
          Chirp
        </button>
        {isAuthenticated && (
          <div className="flex items-center gap-4">
            <button
              onClick={() => onNavigate("home")}
              className="hover:text-blue-200 transition-colors"
            >
              Home
            </button>
            <button
              onClick={() =>
                onNavigate("profile", { userId: String(user!.id) })
              }
              className="hover:text-blue-200 transition-colors"
            >
              Profile
            </button>
            <span className="text-blue-200 text-sm">@{user?.username}</span>
            <button
              onClick={handleLogout}
              className="bg-blue-700 hover:bg-blue-800 px-3 py-1 rounded text-sm transition-colors"
            >
              Logout
            </button>
          </div>
        )}
      </div>
    </nav>
  );
}
