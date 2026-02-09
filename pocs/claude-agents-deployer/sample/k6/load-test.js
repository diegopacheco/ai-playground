import http from "k6/http";
import { check, sleep } from "k6";

export const options = {
  stages: [
    { duration: "30s", target: 10 },
    { duration: "1m", target: 50 },
    { duration: "30s", target: 0 },
  ],
  thresholds: {
    http_req_duration: ["p(95)<500"],
    http_req_failed: ["rate<0.01"],
  },
};

const BASE_URL = "http://localhost:8080";

export default function () {
  const postPayload = JSON.stringify({
    title: `Load Post ${__VU}_${__ITER}`,
    content: `Load test content for iteration ${__ITER}`,
    author: `loaduser_${__VU}`,
  });

  const postRes = http.post(`${BASE_URL}/api/posts`, postPayload, {
    headers: { "Content-Type": "application/json" },
  });
  check(postRes, {
    "create post 201": (r) => r.status === 201,
  });

  const listRes = http.get(`${BASE_URL}/api/posts`);
  check(listRes, {
    "list posts 200": (r) => r.status === 200,
  });

  if (postRes.status === 201) {
    const post = postRes.json();
    const postId = post.id;

    http.get(`${BASE_URL}/api/posts/${postId}`);

    const commentPayload = JSON.stringify({
      content: `Comment ${__ITER}`,
      author: `loaduser_${__VU}`,
    });

    http.post(`${BASE_URL}/api/posts/${postId}/comments`, commentPayload, {
      headers: { "Content-Type": "application/json" },
    });

    http.get(`${BASE_URL}/api/posts/${postId}/comments`);

    const updatePayload = JSON.stringify({
      title: `Updated Load Post ${__VU}_${__ITER}`,
    });

    http.put(`${BASE_URL}/api/posts/${postId}`, updatePayload, {
      headers: { "Content-Type": "application/json" },
    });
  }

  sleep(0.5);
}
