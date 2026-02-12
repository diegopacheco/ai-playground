import { useParams } from "@tanstack/react-router";
import { useUserProfile } from "../hooks/useUsers";
import { useUserPosts } from "../hooks/usePosts";
import { FollowButton } from "../components/FollowButton";
import { Timeline } from "../components/Timeline";
import { getUser } from "../auth";

export function ProfilePage() {
  const { userId } = useParams({ strict: false }) as { userId: string };
  const { data: profile, isLoading: profileLoading } = useUserProfile(userId);
  const { data: posts, isLoading: postsLoading } = useUserPosts(userId);
  const currentUser = getUser();
  const isMe = currentUser?.id === userId;

  if (profileLoading) {
    return (
      <div className="flex justify-center p-8">
        <div className="w-8 h-8 border-4 border-[#1DA1F2] border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  if (!profile) {
    return (
      <div className="p-8 text-center text-gray-500">User not found</div>
    );
  }

  return (
    <div>
      <div className="border-b border-gray-200 p-4">
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center gap-3">
              <div className="w-16 h-16 rounded-full bg-[#1DA1F2] flex items-center justify-center text-white text-2xl font-bold">
                {profile.user.username[0].toUpperCase()}
              </div>
              <div>
                <h2 className="text-xl font-bold text-gray-900">
                  {profile.user.username}
                </h2>
                <p className="text-gray-500">{profile.user.email}</p>
              </div>
            </div>
            <div className="flex gap-5 mt-4">
              <span className="text-gray-700">
                <span className="font-bold">{profile.following_count}</span>{" "}
                <span className="text-gray-500">Following</span>
              </span>
              <span className="text-gray-700">
                <span className="font-bold">{profile.followers_count}</span>{" "}
                <span className="text-gray-500">Followers</span>
              </span>
            </div>
          </div>
          {!isMe && (
            <FollowButton
              userId={userId}
              isFollowing={profile.is_following}
            />
          )}
        </div>
      </div>
      <div className="border-b border-gray-200 p-4">
        <h3 className="font-bold text-gray-900">Posts</h3>
      </div>
      <Timeline posts={posts ?? []} isLoading={postsLoading} />
    </div>
  );
}
