import { useUsers } from "../hooks/useApi";

export default function ProfilePage() {
  const { data: users, isLoading, error } = useUsers();

  if (isLoading) {
    return (
      <div className="flex justify-center py-20">
        <div className="text-gray-500 text-lg">Loading users...</div>
      </div>
    );
  }

  if (error || !users || users.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8 max-w-lg mx-auto">
        <div className="text-center">
          <div className="w-24 h-24 bg-indigo-100 rounded-full mx-auto mb-4 flex items-center justify-center">
            <span className="text-3xl text-indigo-600 font-bold">?</span>
          </div>
          <h1 className="text-2xl font-bold text-gray-900 mb-2">
            No Users
          </h1>
          <p className="text-gray-500">
            No users found. Create posts to get started.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-lg mx-auto">
      <h1 className="text-3xl font-bold text-gray-900 mb-8">Users</h1>
      <div className="space-y-4">
        {users.map((user) => (
          <div
            key={user.id}
            className="bg-white rounded-lg shadow-sm border border-gray-200 p-6"
          >
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 bg-indigo-100 rounded-full flex items-center justify-center">
                <span className="text-lg text-indigo-600 font-bold">
                  {user.name.charAt(0).toUpperCase()}
                </span>
              </div>
              <div>
                <h2 className="text-lg font-semibold text-gray-900">
                  {user.name}
                </h2>
                <p className="text-gray-500 text-sm">{user.email}</p>
              </div>
            </div>
            <div className="mt-3 text-xs text-gray-400">
              Joined {new Date(user.createdAt).toLocaleDateString()}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
