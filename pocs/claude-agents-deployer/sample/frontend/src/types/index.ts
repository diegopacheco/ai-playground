export interface Post {
  id: number;
  title: string;
  content: string;
  excerpt: string;
  author: string;
  createdAt: string;
  updatedAt: string;
}

export interface Comment {
  id: number;
  postId: number;
  author: string;
  content: string;
  createdAt: string;
}

export interface User {
  id: number;
  name: string;
  email: string;
  bio: string;
  avatarUrl: string;
  createdAt: string;
}

export interface CreatePostPayload {
  title: string;
  content: string;
  excerpt: string;
  author: string;
}

export interface UpdatePostPayload {
  title: string;
  content: string;
  excerpt: string;
}

export interface CreateCommentPayload {
  postId: number;
  author: string;
  content: string;
}
