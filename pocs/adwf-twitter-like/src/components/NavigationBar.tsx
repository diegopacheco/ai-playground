import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';

export const NavigationBar = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <nav className="bg-blue-500 text-white p-4 sticky top-0 z-50 shadow-md">
      <div className="container mx-auto flex justify-between items-center">
        <div className="flex items-center space-x-6">
          <Link to="/" className="text-2xl font-bold hover:text-blue-100">
            Twitter Clone
          </Link>
          <Link to="/" className="hover:text-blue-100">
            Home
          </Link>
          {user && (
            <Link to={`/profile/${user.id}`} className="hover:text-blue-100">
              Profile
            </Link>
          )}
        </div>
        <div className="flex items-center space-x-4">
          {user && (
            <>
              <span className="text-sm">@{user.username}</span>
              <button
                onClick={handleLogout}
                className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded"
              >
                Logout
              </button>
            </>
          )}
        </div>
      </div>
    </nav>
  );
};
