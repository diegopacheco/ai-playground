import http from "k6/http";
import { check, sleep } from "k6";

export const options = {
  stages: [
    { duration: "10s", target: 10 },
    { duration: "20s", target: 20 },
    { duration: "10s", target: 0 },
  ],
  thresholds: {
    http_req_duration: ["p(95)<2000"],
    http_req_failed: ["rate<0.1"],
  },
};

const BASE_URL = "http://localhost:3000";

function randomString(len) {
  let chars = "abcdefghijklmnopqrstuvwxyz0123456789";
  let result = "";
  for (let i = 0; i < len; i++) {
    result += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return result;
}

export default function () {
  let suffix = randomString(12);
  let username = `k6user_${suffix}`;
  let email = `${username}@test.com`;
  let password = "k6password";

  let registerRes = http.post(
    `${BASE_URL}/api/auth/register`,
    JSON.stringify({
      username: username,
      email: email,
      password: password,
    }),
    { headers: { "Content-Type": "application/json" } }
  );

  check(registerRes, {
    "register status 200": (r) => r.status === 200,
  });

  if (registerRes.status !== 200) {
    return;
  }

  let authData = JSON.parse(registerRes.body);
  let token = authData.token;
  let userId = authData.user.id;

  let authHeaders = {
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
  };

  let loginRes = http.post(
    `${BASE_URL}/api/auth/login`,
    JSON.stringify({ email: email, password: password }),
    { headers: { "Content-Type": "application/json" } }
  );

  check(loginRes, {
    "login status 200": (r) => r.status === 200,
  });

  let meRes = http.get(`${BASE_URL}/api/auth/me`, authHeaders);

  check(meRes, {
    "me status 200": (r) => r.status === 200,
  });

  let createRes = http.post(
    `${BASE_URL}/api/posts`,
    JSON.stringify({ content: `K6 stress test post ${suffix}` }),
    authHeaders
  );

  check(createRes, {
    "create post status 200": (r) => r.status === 200,
  });

  let postId = "";
  if (createRes.status === 200) {
    postId = JSON.parse(createRes.body).id;
  }

  let timelineRes = http.get(`${BASE_URL}/api/posts`, authHeaders);

  check(timelineRes, {
    "timeline status 200": (r) => r.status === 200,
  });

  if (postId) {
    let getPostRes = http.get(
      `${BASE_URL}/api/posts/${postId}`,
      authHeaders
    );

    check(getPostRes, {
      "get post status 200": (r) => r.status === 200,
    });

    let likeRes = http.post(
      `${BASE_URL}/api/posts/${postId}/like`,
      null,
      authHeaders
    );

    check(likeRes, {
      "like post status 200": (r) => r.status === 200,
    });

    let unlikeRes = http.del(
      `${BASE_URL}/api/posts/${postId}/like`,
      null,
      authHeaders
    );

    check(unlikeRes, {
      "unlike post status 200": (r) => r.status === 200,
    });
  }

  let profileRes = http.get(
    `${BASE_URL}/api/users/${userId}`,
    authHeaders
  );

  check(profileRes, {
    "profile status 200": (r) => r.status === 200,
  });

  let userPostsRes = http.get(
    `${BASE_URL}/api/users/${userId}/posts`,
    authHeaders
  );

  check(userPostsRes, {
    "user posts status 200": (r) => r.status === 200,
  });

  sleep(0.5);
}
