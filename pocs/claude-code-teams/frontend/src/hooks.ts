import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { fetchTweets, createTweet, deleteTweet, likeTweet } from "./api";

export function useTweets() {
  return useQuery({
    queryKey: ["tweets"],
    queryFn: fetchTweets,
    refetchInterval: 5000,
  });
}

export function useCreateTweet() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ username, content }: { username: string; content: string }) =>
      createTweet(username, content),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["tweets"] });
    },
  });
}

export function useDeleteTweet() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => deleteTweet(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["tweets"] });
    },
  });
}

export function useLikeTweet() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => likeTweet(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["tweets"] });
    },
  });
}
