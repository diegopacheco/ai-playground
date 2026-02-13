import { PostComposer } from "../components/PostComposer";
import { Feed } from "../components/Feed";
import { getFeed } from "../api/posts";

interface HomePageProps {
  onNavigate: (page: string, params?: Record<string, string>) => void;
}

export function HomePage({ onNavigate }: HomePageProps) {
  return (
    <div className="max-w-2xl mx-auto px-4 py-6 space-y-4">
      <PostComposer />
      <Feed
        queryKey={["feed"]}
        fetchFn={getFeed}
        onAuthorClick={(userId) =>
          onNavigate("profile", { userId: String(userId) })
        }
        onPostClick={(postId) =>
          onNavigate("postDetail", { postId: String(postId) })
        }
      />
    </div>
  );
}
