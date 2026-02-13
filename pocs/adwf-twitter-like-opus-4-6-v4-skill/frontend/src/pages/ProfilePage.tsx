import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { getUser, getUserFollowers, getUserFollowing } from "../api/users";
import { getUserPosts } from "../api/posts";
import { Feed } from "../components/Feed";
import { FollowersList } from "../components/FollowersList";
import { useAuth } from "../hooks/useAuth";

interface ProfilePageProps {
  userId: number;
  onNavigate: (page: string, params?: Record<string, string>) => void;
}

export function ProfilePage({ userId, onNavigate }: ProfilePageProps) {
  const [activeTab, setActiveTab] = useState<"posts" | "followers" | "following">("posts");
  const { user: currentUser } = useAuth();

  const { data: profileUser, isLoading } = useQuery({
    queryKey: ["user", userId],
    queryFn: () => getUser(userId),
  });

  const { data: followers } = useQuery({
    queryKey: ["followers", userId],
    queryFn: () => getUserFollowers(userId),
  });

  const { data: following } = useQuery({
    queryKey: ["following", userId],
    queryFn: () => getUserFollowing(userId),
  });

  if (isLoading) {
    return (
      <div className="flex justify-center py-12">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500" />
      </div>
    );
  }

  if (!profileUser) {
    return (
      <div className="text-center py-12 text-gray-500">
        User not found.
      </div>
    );
  }

  const isOwnProfile = currentUser?.id === profileUser.id;

  const tabs = [
    { key: "posts" as const, label: "Posts" },
    { key: "followers" as const, label: `Followers (${followers?.length ?? 0})` },
    { key: "following" as const, label: `Following (${following?.length ?? 0})` },
  ];

  return (
    <div className="max-w-2xl mx-auto px-4 py-6">
      <div className="bg-white border border-gray-200 rounded-lg p-6 mb-4">
        <div className="flex items-center gap-4">
          <div className="w-16 h-16 bg-blue-500 rounded-full flex items-center justify-center text-white text-2xl font-bold">
            {profileUser.username[0].toUpperCase()}
          </div>
          <div>
            <h1 className="text-2xl font-bold text-gray-900">
              @{profileUser.username}
            </h1>
            <p className="text-gray-500 text-sm">
              Joined {new Date(profileUser.created_at).toLocaleDateString()}
            </p>
            {isOwnProfile && (
              <span className="text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full">
                You
              </span>
            )}
          </div>
        </div>
      </div>

      <div className="flex border-b border-gray-200 mb-4">
        {tabs.map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            className={`flex-1 py-3 text-center font-semibold text-sm transition-colors ${
              activeTab === tab.key
                ? "text-blue-600 border-b-2 border-blue-600"
                : "text-gray-500 hover:text-gray-700"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {activeTab === "posts" && (
        <Feed
          queryKey={["posts", "user", String(userId)]}
          fetchFn={(page) => getUserPosts(userId, page)}
          onAuthorClick={(uid) =>
            onNavigate("profile", { userId: String(uid) })
          }
          onPostClick={(postId) =>
            onNavigate("postDetail", { postId: String(postId) })
          }
        />
      )}

      {activeTab === "followers" && (
        <FollowersList
          userId={userId}
          type="followers"
          onUserClick={(uid) =>
            onNavigate("profile", { userId: String(uid) })
          }
        />
      )}

      {activeTab === "following" && (
        <FollowersList
          userId={userId}
          type="following"
          onUserClick={(uid) =>
            onNavigate("profile", { userId: String(uid) })
          }
        />
      )}
    </div>
  );
}
