import { PostComposer } from "../components/PostComposer";
import { Timeline } from "../components/Timeline";
import { useTimeline } from "../hooks/usePosts";

export function HomePage() {
  const { data: posts, isLoading } = useTimeline();

  return (
    <div>
      <div className="border-b border-gray-200 p-4">
        <h2 className="text-xl font-bold text-gray-900">Home</h2>
      </div>
      <PostComposer />
      <Timeline posts={posts ?? []} isLoading={isLoading} />
    </div>
  );
}
