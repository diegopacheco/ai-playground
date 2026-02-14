import { Link } from "react-router-dom";
import { useAuth } from "../hooks/useAuth";

export function NavBar() {
  const { user, logout, isAuthenticated } = useAuth();

  if (!isAuthenticated) return null;

  return (
    <nav className="bg-blue-600 text-white shadow-lg">
      <div className="max-w-3xl mx-auto px-4 py-3 flex items-center justify-between">
        <Link to="/" className="text-xl font-bold">
          Chirper
        </Link>
        <div className="flex items-center gap-4">
          <Link to="/" className="hover:text-blue-200">
            Home
          </Link>
          {user && (
            <Link to={`/profile/${user.id}`} className="hover:text-blue-200">
              Profile
            </Link>
          )}
          <button
            onClick={logout}
            className="bg-blue-800 px-3 py-1 rounded hover:bg-blue-900"
          >
            Logout
          </button>
        </div>
      </div>
    </nav>
  );
}
