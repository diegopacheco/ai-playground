import http from "k6/http";
import { check, sleep } from "k6";

export const options = {
  stages: [
    { duration: "30s", target: 50 },
    { duration: "1m", target: 100 },
    { duration: "1m", target: 150 },
    { duration: "30s", target: 0 },
  ],
  thresholds: {
    http_req_duration: ["p(95)<1000"],
    http_req_failed: ["rate<0.05"],
  },
};

const BASE_URL = "http://localhost:8080";

export default function () {
  const postPayload = JSON.stringify({
    title: `Stress ${__VU}_${__ITER}`,
    content: `Stress test content ${__VU}_${__ITER}`,
    author: `stressuser_${__VU}`,
  });

  const postRes = http.post(`${BASE_URL}/api/posts`, postPayload, {
    headers: { "Content-Type": "application/json" },
  });
  check(postRes, {
    "post created": (r) => r.status === 201,
  });

  http.get(`${BASE_URL}/api/posts`);

  if (postRes.status === 201) {
    const post = postRes.json();
    const postId = post.id;

    http.get(`${BASE_URL}/api/posts/${postId}`);

    for (let i = 0; i < 3; i++) {
      const commentPayload = JSON.stringify({
        content: `Stress comment ${i}`,
        author: `stressuser_${__VU}`,
      });
      http.post(`${BASE_URL}/api/posts/${postId}/comments`, commentPayload, {
        headers: { "Content-Type": "application/json" },
      });
    }

    http.get(`${BASE_URL}/api/posts/${postId}/comments`);

    http.del(`${BASE_URL}/api/posts/${postId}`);
  }

  sleep(0.3);
}
