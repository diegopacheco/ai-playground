import { useState } from "react";
import { Link } from "@tanstack/react-router";
import { useLogin } from "../hooks/useAuth";

export function LoginPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const login = useLogin();

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    login.mutate({ email, password });
  }

  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center">
      <div className="bg-white p-8 rounded-2xl shadow-lg w-full max-w-md">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-[#1DA1F2]">TwitterClone</h1>
          <p className="text-gray-500 mt-2">Sign in to your account</p>
        </div>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label
              htmlFor="email"
              className="block text-sm font-medium text-gray-700 mb-1"
            >
              Email
            </label>
            <input
              id="email"
              type="email"
              data-testid="login-email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#1DA1F2] focus:border-transparent"
              placeholder="you@mail.com"
            />
          </div>
          <div>
            <label
              htmlFor="password"
              className="block text-sm font-medium text-gray-700 mb-1"
            >
              Password
            </label>
            <input
              id="password"
              type="password"
              data-testid="login-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#1DA1F2] focus:border-transparent"
              placeholder="Your password"
            />
          </div>
          {login.isError && (
            <p className="text-red-500 text-sm">{login.error.message}</p>
          )}
          <button
            type="submit"
            data-testid="login-submit"
            disabled={login.isPending}
            className="w-full bg-[#1DA1F2] text-white py-2.5 rounded-full font-bold hover:bg-[#1a91da] disabled:opacity-50 cursor-pointer"
          >
            {login.isPending ? "Signing in..." : "Sign in"}
          </button>
        </form>
        <p className="text-center mt-6 text-gray-500">
          Don't have an account?{" "}
          <Link to="/register" className="text-[#1DA1F2] hover:underline">
            Register
          </Link>
        </p>
      </div>
    </div>
  );
}
