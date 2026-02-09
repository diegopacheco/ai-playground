import {
  createRouter,
  createRoute,
  createRootRoute,
  Link,
  Outlet,
} from "@tanstack/react-router";
import HomePage from "./pages/HomePage";
import PostDetailPage from "./pages/PostDetailPage";
import CreatePostPage from "./pages/CreatePostPage";
import EditPostPage from "./pages/EditPostPage";
import ProfilePage from "./pages/ProfilePage";

const rootRoute = createRootRoute({
  component: () => (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-5xl mx-auto px-4 py-4 flex items-center justify-between">
          <Link to="/" className="text-2xl font-bold text-indigo-600">
            Blog Platform
          </Link>
          <div className="flex gap-6 items-center">
            <Link
              to="/"
              className="text-gray-600 hover:text-indigo-600 font-medium transition-colors"
            >
              Home
            </Link>
            <Link
              to="/posts/create"
              className="text-gray-600 hover:text-indigo-600 font-medium transition-colors"
            >
              New Post
            </Link>
            <Link
              to="/profile"
              className="text-gray-600 hover:text-indigo-600 font-medium transition-colors"
            >
              Profile
            </Link>
          </div>
        </div>
      </nav>
      <main className="max-w-5xl mx-auto px-4 py-8">
        <Outlet />
      </main>
    </div>
  ),
});

const indexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  component: HomePage,
});

const postDetailRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/posts/$postId",
  component: () => {
    const { postId } = postDetailRoute.useParams();
    return <PostDetailPage postId={postId} />;
  },
});

const createPostRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/posts/create",
  component: CreatePostPage,
});

const editPostRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/posts/$postId/edit",
  component: () => {
    const { postId } = editPostRoute.useParams();
    return <EditPostPage postId={postId} />;
  },
});

const profileRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/profile",
  component: ProfilePage,
});

const routeTree = rootRoute.addChildren([
  indexRoute,
  createPostRoute,
  postDetailRoute,
  editPostRoute,
  profileRoute,
]);

export const router = createRouter({ routeTree });

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}
