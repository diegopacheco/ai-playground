import http from "k6/http";
import { check, sleep } from "k6";

export const options = {
  vus: 2,
  duration: "30s",
  thresholds: {
    http_req_duration: ["p(95)<500"],
    http_req_failed: ["rate<0.01"],
  },
};

const BASE_URL = "http://localhost:8080";

export default function () {
  const userPayload = JSON.stringify({
    name: `user_${__VU}_${__ITER}`,
    email: `user_${__VU}_${__ITER}@test.com`,
  });

  const userRes = http.post(`${BASE_URL}/api/users`, userPayload, {
    headers: { "Content-Type": "application/json" },
  });
  check(userRes, {
    "create user status 201": (r) => r.status === 201,
  });

  const listUsersRes = http.get(`${BASE_URL}/api/users`);
  check(listUsersRes, {
    "list users status 200": (r) => r.status === 200,
  });

  const postPayload = JSON.stringify({
    title: `Post ${__VU}_${__ITER}`,
    content: `Content for post ${__VU}_${__ITER}`,
    author: `user_${__VU}_${__ITER}`,
  });

  const postRes = http.post(`${BASE_URL}/api/posts`, postPayload, {
    headers: { "Content-Type": "application/json" },
  });
  check(postRes, {
    "create post status 201": (r) => r.status === 201,
  });

  const listPostsRes = http.get(`${BASE_URL}/api/posts`);
  check(listPostsRes, {
    "list posts status 200": (r) => r.status === 200,
  });

  if (postRes.status === 201) {
    const post = postRes.json();
    const postId = post.id;

    const getPostRes = http.get(`${BASE_URL}/api/posts/${postId}`);
    check(getPostRes, {
      "get post status 200": (r) => r.status === 200,
    });

    const commentPayload = JSON.stringify({
      content: `Comment on ${postId}`,
      author: `user_${__VU}`,
    });

    const commentRes = http.post(`${BASE_URL}/api/posts/${postId}/comments`, commentPayload, {
      headers: { "Content-Type": "application/json" },
    });
    check(commentRes, {
      "create comment status 201": (r) => r.status === 201,
    });

    const listCommentsRes = http.get(`${BASE_URL}/api/posts/${postId}/comments`);
    check(listCommentsRes, {
      "list comments status 200": (r) => r.status === 200,
    });
  }

  sleep(1);
}
