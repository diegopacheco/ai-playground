import { AuthResponse, LoginCredentials, RegisterCredentials, User, Tweet, Comment } from '@/types';

const API_BASE_URL = 'http://localhost:8000/api';

const getAuthToken = (): string | null => {
  return localStorage.getItem('token');
};

const makeRequest = async <T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> => {
  const token = getAuthToken();
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    ...options.headers,
  };

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    ...options,
    headers,
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(error || `HTTP error! status: ${response.status}`);
  }

  return response.json();
};

export const authApi = {
  login: (credentials: LoginCredentials) =>
    makeRequest<AuthResponse>('/auth/login', {
      method: 'POST',
      body: JSON.stringify(credentials),
    }),

  register: (credentials: RegisterCredentials) =>
    makeRequest<AuthResponse>('/auth/register', {
      method: 'POST',
      body: JSON.stringify(credentials),
    }),

  logout: () =>
    makeRequest<void>('/auth/logout', {
      method: 'POST',
    }),
};

export const usersApi = {
  getUser: (id: number) => makeRequest<User>(`/users/${id}`),

  updateUser: (id: number, data: Partial<User>) =>
    makeRequest<User>(`/users/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    }),

  getFollowers: (id: number) => makeRequest<User[]>(`/users/${id}/followers`),

  getFollowing: (id: number) => makeRequest<User[]>(`/users/${id}/following`),

  follow: (id: number) =>
    makeRequest<void>(`/users/${id}/follow`, {
      method: 'POST',
    }),

  unfollow: (id: number) =>
    makeRequest<void>(`/users/${id}/follow`, {
      method: 'DELETE',
    }),
};

export const tweetsApi = {
  createTweet: (content: string) =>
    makeRequest<Tweet>('/tweets', {
      method: 'POST',
      body: JSON.stringify({ content }),
    }),

  getTweet: (id: number) => makeRequest<Tweet>(`/tweets/${id}`),

  deleteTweet: (id: number) =>
    makeRequest<void>(`/tweets/${id}`, {
      method: 'DELETE',
    }),

  getFeed: () => makeRequest<Tweet[]>('/tweets/feed'),

  getUserTweets: (userId: number) =>
    makeRequest<Tweet[]>(`/tweets/user/${userId}`),

  likeTweet: (id: number) =>
    makeRequest<void>(`/tweets/${id}/like`, {
      method: 'POST',
    }),

  unlikeTweet: (id: number) =>
    makeRequest<void>(`/tweets/${id}/like`, {
      method: 'DELETE',
    }),

  retweetTweet: (id: number) =>
    makeRequest<void>(`/tweets/${id}/retweet`, {
      method: 'POST',
    }),

  unretweetTweet: (id: number) =>
    makeRequest<void>(`/tweets/${id}/retweet`, {
      method: 'DELETE',
    }),
};

export const commentsApi = {
  addComment: (tweetId: number, content: string) =>
    makeRequest<Comment>(`/tweets/${tweetId}/comments`, {
      method: 'POST',
      body: JSON.stringify({ content }),
    }),

  getComments: (tweetId: number) =>
    makeRequest<Comment[]>(`/tweets/${tweetId}/comments`),

  deleteComment: (id: number) =>
    makeRequest<void>(`/comments/${id}`, {
      method: 'DELETE',
    }),
};
