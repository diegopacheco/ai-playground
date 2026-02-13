import { useState, useCallback } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AuthProvider } from "./context/AuthContext";
import { useAuth } from "./hooks/useAuth";
import { Navbar } from "./components/Navbar";
import { LoginPage } from "./pages/LoginPage";
import { RegisterPage } from "./pages/RegisterPage";
import { HomePage } from "./pages/HomePage";
import { ProfilePage } from "./pages/ProfilePage";
import { PostDetailPage } from "./pages/PostDetailPage";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      staleTime: 30000,
    },
  },
});

interface RouteState {
  page: string;
  params: Record<string, string>;
}

function Router() {
  const { isAuthenticated } = useAuth();
  const [route, setRoute] = useState<RouteState>({
    page: isAuthenticated ? "home" : "login",
    params: {},
  });

  const onNavigate = useCallback(
    (page: string, params?: Record<string, string>) => {
      setRoute({ page, params: params ?? {} });
    },
    []
  );

  if (!isAuthenticated && route.page !== "login" && route.page !== "register") {
    return <LoginPage onNavigate={onNavigate} />;
  }

  if (route.page === "login") {
    return <LoginPage onNavigate={onNavigate} />;
  }

  if (route.page === "register") {
    return <RegisterPage onNavigate={onNavigate} />;
  }

  return (
    <div className="min-h-screen bg-gray-100">
      <Navbar onNavigate={onNavigate} />
      {route.page === "home" && <HomePage onNavigate={onNavigate} />}
      {route.page === "profile" && (
        <ProfilePage
          userId={Number(route.params.userId)}
          onNavigate={onNavigate}
        />
      )}
      {route.page === "postDetail" && (
        <PostDetailPage
          postId={Number(route.params.postId)}
          onNavigate={onNavigate}
        />
      )}
    </div>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <Router />
      </AuthProvider>
    </QueryClientProvider>
  );
}
