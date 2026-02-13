import http from "k6/http";
import { check, sleep } from "k6";

export const options = {
  stages: [
    { duration: "10s", target: 10 },
    { duration: "20s", target: 10 },
    { duration: "5s", target: 0 },
  ],
  thresholds: {
    http_req_duration: ["p(95)<2000"],
    http_req_failed: ["rate<0.1"],
  },
};

const BASE_URL = "http://localhost:8080";

export default function () {
  const uniqueId = `${__VU}_${__ITER}_${Date.now()}`;
  const registerPayload = JSON.stringify({
    username: `k6user_${uniqueId}`,
    email: `k6user_${uniqueId}@test.com`,
    password: "password123",
  });
  const registerRes = http.post(`${BASE_URL}/api/users`, registerPayload, {
    headers: { "Content-Type": "application/json" },
  });
  check(registerRes, {
    "register status is 201": (r) => r.status === 201,
  });
  const loginPayload = JSON.stringify({
    username: `k6user_${uniqueId}`,
    password: "password123",
  });
  const loginRes = http.post(`${BASE_URL}/api/users/login`, loginPayload, {
    headers: { "Content-Type": "application/json" },
  });
  check(loginRes, {
    "login status is 200": (r) => r.status === 200,
  });
  let token = "";
  try {
    token = JSON.parse(loginRes.body).token;
  } catch (e) {
    return;
  }
  const authHeaders = {
    "Content-Type": "application/json",
    Authorization: `Bearer ${token}`,
  };
  const postPayload = JSON.stringify({
    content: `K6 load test post ${uniqueId}`,
  });
  const postRes = http.post(`${BASE_URL}/api/posts`, postPayload, {
    headers: authHeaders,
  });
  check(postRes, {
    "create post status is 200 or 201": (r) => r.status === 200 || r.status === 201,
  });
  const postsRes = http.get(`${BASE_URL}/api/posts`);
  check(postsRes, {
    "get posts status is 200": (r) => r.status === 200,
  });
  const feedRes = http.get(`${BASE_URL}/api/feed`, {
    headers: authHeaders,
  });
  check(feedRes, {
    "get feed status is 200": (r) => r.status === 200,
  });
  sleep(0.5);
}
