import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { loginUser } from "../api/users";
import { useAuth } from "../hooks/useAuth";

interface LoginPageProps {
  onNavigate: (page: string) => void;
}

export function LoginPage({ onNavigate }: LoginPageProps) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const { login } = useAuth();

  const mutation = useMutation({
    mutationFn: loginUser,
    onSuccess: (data) => {
      login(data.token, data.user);
      onNavigate("home");
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    mutation.mutate({ username, password });
  };

  return (
    <div className="min-h-screen bg-gray-100 flex items-center justify-center">
      <div className="bg-white p-8 rounded-lg shadow-md w-full max-w-md">
        <h1 className="text-3xl font-bold text-center text-blue-600 mb-6">
          Chirp
        </h1>
        <h2 className="text-xl font-semibold text-gray-800 mb-4">Login</h2>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Username
            </label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              required
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Password
            </label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              required
            />
          </div>
          <button
            type="submit"
            disabled={mutation.isPending}
            className="w-full bg-blue-500 hover:bg-blue-600 disabled:bg-blue-300 text-white py-2 rounded-md font-semibold transition-colors"
          >
            {mutation.isPending ? "Logging in..." : "Login"}
          </button>
          {mutation.isError && (
            <p className="text-red-500 text-sm text-center">
              Login failed. Check your credentials.
            </p>
          )}
        </form>
        <p className="text-center mt-4 text-gray-600">
          Don&apos;t have an account?{" "}
          <button
            onClick={() => onNavigate("register")}
            className="text-blue-500 hover:underline font-semibold"
          >
            Register
          </button>
        </p>
      </div>
    </div>
  );
}
