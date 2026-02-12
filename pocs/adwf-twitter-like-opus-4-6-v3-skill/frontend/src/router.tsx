import {
  createRouter,
  createRoute,
  createRootRoute,
  redirect,
  Outlet,
} from "@tanstack/react-router";
import { isAuthenticated } from "./auth";
import { Navbar } from "./components/Navbar";
import { LoginPage } from "./pages/LoginPage";
import { RegisterPage } from "./pages/RegisterPage";
import { HomePage } from "./pages/HomePage";
import { ProfilePage } from "./pages/ProfilePage";
import { PostDetailPage } from "./pages/PostDetailPage";

const rootRoute = createRootRoute({
  component: () => <Outlet />,
});

const authLayout = createRoute({
  id: "auth",
  getParentRoute: () => rootRoute,
  beforeLoad: () => {
    if (!isAuthenticated()) {
      throw redirect({ to: "/login" });
    }
  },
  component: () => (
    <div className="min-h-screen bg-gray-50">
      <Navbar />
      <main className="max-w-2xl mx-auto bg-white min-h-[calc(100vh-56px)] border-x border-gray-200">
        <Outlet />
      </main>
    </div>
  ),
});

const loginRoute = createRoute({
  path: "/login",
  getParentRoute: () => rootRoute,
  component: LoginPage,
});

const registerRoute = createRoute({
  path: "/register",
  getParentRoute: () => rootRoute,
  component: RegisterPage,
});

const homeRoute = createRoute({
  path: "/",
  getParentRoute: () => authLayout,
  component: HomePage,
});

const profileRoute = createRoute({
  path: "/profile/$userId",
  getParentRoute: () => authLayout,
  component: ProfilePage,
});

const postDetailRoute = createRoute({
  path: "/post/$postId",
  getParentRoute: () => authLayout,
  component: PostDetailPage,
});

const routeTree = rootRoute.addChildren([
  loginRoute,
  registerRoute,
  authLayout.addChildren([homeRoute, profileRoute, postDetailRoute]),
]);

export const router = createRouter({ routeTree });

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}
