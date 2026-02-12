import { Link } from "@tanstack/react-router";
import { getUser } from "../auth";
import { useLogout } from "../hooks/useAuth";

export function Navbar() {
  const user = getUser();
  const logout = useLogout();

  return (
    <nav className="bg-white border-b border-gray-200 sticky top-0 z-50">
      <div className="max-w-2xl mx-auto px-4 flex items-center justify-between h-14">
        <Link to="/" data-testid="nav-home" className="text-xl font-bold text-[#1DA1F2]">
          TwitterClone
        </Link>
        <div className="flex items-center gap-4">
          <Link
            to="/"
            className="text-gray-700 hover:text-[#1DA1F2] font-medium"
          >
            Home
          </Link>
          {user && (
            <Link
              to="/profile/$userId"
              params={{ userId: user.id }}
              className="text-gray-700 hover:text-[#1DA1F2] font-medium"
            >
              Profile
            </Link>
          )}
          <button
            data-testid="nav-logout"
            onClick={logout}
            className="text-gray-500 hover:text-red-500 font-medium cursor-pointer"
          >
            Logout
          </button>
        </div>
      </div>
    </nav>
  );
}
