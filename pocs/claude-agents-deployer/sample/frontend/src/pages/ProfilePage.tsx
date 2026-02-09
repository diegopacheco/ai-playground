import { useCurrentUser } from "../hooks/useApi";

export default function ProfilePage() {
  const { data: user, isLoading, error } = useCurrentUser();

  if (isLoading) {
    return (
      <div className="flex justify-center py-20">
        <div className="text-gray-500 text-lg">Loading profile...</div>
      </div>
    );
  }

  if (error || !user) {
    return (
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8 max-w-lg mx-auto">
        <div className="text-center">
          <div className="w-24 h-24 bg-indigo-100 rounded-full mx-auto mb-4 flex items-center justify-center">
            <span className="text-3xl text-indigo-600 font-bold">?</span>
          </div>
          <h1 className="text-2xl font-bold text-gray-900 mb-2">
            Profile Unavailable
          </h1>
          <p className="text-gray-500">
            Could not load user profile. Make sure the API server is running.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-lg mx-auto">
      <h1 className="text-3xl font-bold text-gray-900 mb-8">User Profile</h1>
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8">
        <div className="flex items-center gap-6 mb-6">
          {user.avatarUrl ? (
            <img
              src={user.avatarUrl}
              alt={user.name}
              className="w-20 h-20 rounded-full object-cover"
            />
          ) : (
            <div className="w-20 h-20 bg-indigo-100 rounded-full flex items-center justify-center">
              <span className="text-2xl text-indigo-600 font-bold">
                {user.name.charAt(0).toUpperCase()}
              </span>
            </div>
          )}
          <div>
            <h2 className="text-xl font-semibold text-gray-900">
              {user.name}
            </h2>
            <p className="text-gray-500">{user.email}</p>
          </div>
        </div>
        {user.bio && (
          <div className="border-t border-gray-100 pt-4">
            <h3 className="text-sm font-medium text-gray-500 mb-2">Bio</h3>
            <p className="text-gray-700 leading-relaxed">{user.bio}</p>
          </div>
        )}
        <div className="border-t border-gray-100 pt-4 mt-4">
          <h3 className="text-sm font-medium text-gray-500 mb-2">
            Member since
          </h3>
          <p className="text-gray-700">
            {new Date(user.createdAt).toLocaleDateString()}
          </p>
        </div>
      </div>
    </div>
  );
}
