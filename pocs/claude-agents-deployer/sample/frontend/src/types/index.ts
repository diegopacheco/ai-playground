export interface Post {
  id: string;
  title: string;
  content: string;
  author: string;
  createdAt: string;
  updatedAt: string;
}

export interface Comment {
  id: string;
  postId: string;
  author: string;
  content: string;
  createdAt: string;
}

export interface User {
  id: string;
  name: string;
  email: string;
  createdAt: string;
}

export interface CreatePostPayload {
  title: string;
  content: string;
  author: string;
}

export interface UpdatePostPayload {
  title: string;
  content: string;
}

export interface CreateCommentPayload {
  author: string;
  content: string;
}
