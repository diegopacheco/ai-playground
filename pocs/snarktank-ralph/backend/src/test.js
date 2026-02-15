const { db, initializeDatabase } = require('./db');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const { JWT_SECRET } = require('./auth');

let passed = 0;
let failed = 0;

function assert(condition, message) {
  if (condition) {
    console.log(`PASS: ${message}`);
    passed++;
  } else {
    console.log(`FAIL: ${message}`);
    failed++;
  }
}

initializeDatabase();

db.exec("DELETE FROM likes; DELETE FROM follows; DELETE FROM snarks; DELETE FROM users;");

const tables = db.prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").all();
const tableNames = tables.map(t => t.name);

assert(tableNames.includes('users'), 'users table exists');
assert(tableNames.includes('snarks'), 'snarks table exists');
assert(tableNames.includes('likes'), 'likes table exists');
assert(tableNames.includes('follows'), 'follows table exists');

const userColumns = db.prepare("PRAGMA table_info(users)").all().map(c => c.name);
assert(userColumns.includes('id'), 'users has id column');
assert(userColumns.includes('username'), 'users has username column');
assert(userColumns.includes('display_name'), 'users has display_name column');
assert(userColumns.includes('password'), 'users has password column');
assert(userColumns.includes('bio'), 'users has bio column');
assert(userColumns.includes('created_at'), 'users has created_at column');

const snarkColumns = db.prepare("PRAGMA table_info(snarks)").all().map(c => c.name);
assert(snarkColumns.includes('id'), 'snarks has id column');
assert(snarkColumns.includes('user_id'), 'snarks has user_id column');
assert(snarkColumns.includes('content'), 'snarks has content column');
assert(snarkColumns.includes('parent_id'), 'snarks has parent_id column');
assert(snarkColumns.includes('created_at'), 'snarks has created_at column');

const likeColumns = db.prepare("PRAGMA table_info(likes)").all().map(c => c.name);
assert(likeColumns.includes('user_id'), 'likes has user_id column');
assert(likeColumns.includes('snark_id'), 'likes has snark_id column');

const followColumns = db.prepare("PRAGMA table_info(follows)").all().map(c => c.name);
assert(followColumns.includes('follower_id'), 'follows has follower_id column');
assert(followColumns.includes('following_id'), 'follows has following_id column');

const hashedPassword = bcrypt.hashSync('testpass123', 10);
const result = db.prepare('INSERT INTO users (username, display_name, password) VALUES (?, ?, ?)').run('testuser', 'Test User', hashedPassword);
assert(result.lastInsertRowid > 0, 'user registration inserts into database');

const user = db.prepare('SELECT * FROM users WHERE username = ?').get('testuser');
assert(user !== undefined, 'registered user can be found by username');
assert(user.display_name === 'Test User', 'user display name stored correctly');
assert(bcrypt.compareSync('testpass123', user.password), 'password is hashed and verifiable');
assert(!bcrypt.compareSync('wrongpassword', user.password), 'wrong password does not match');

const token = jwt.sign({ id: user.id, username: user.username }, JWT_SECRET, { expiresIn: '7d' });
const decoded = jwt.verify(token, JWT_SECRET);
assert(decoded.id === user.id, 'JWT token contains user id');
assert(decoded.username === 'testuser', 'JWT token contains username');

let duplicateError = false;
try {
  db.prepare('INSERT INTO users (username, display_name, password) VALUES (?, ?, ?)').run('testuser', 'Another User', hashedPassword);
} catch (e) {
  duplicateError = true;
}
assert(duplicateError, 'duplicate username is rejected');

const hashedPassword2 = bcrypt.hashSync('pass456', 10);
const result2 = db.prepare('INSERT INTO users (username, display_name, password) VALUES (?, ?, ?)').run('testuser2', 'Test User 2', hashedPassword2);
assert(result2.lastInsertRowid > 0, 'second user can register with different username');

const snarkResult = db.prepare('INSERT INTO snarks (user_id, content) VALUES (?, ?)').run(user.id, 'Hello SnarkTank!');
assert(snarkResult.lastInsertRowid > 0, 'snark can be inserted');

const snark = db.prepare('SELECT * FROM snarks WHERE id = ?').get(snarkResult.lastInsertRowid);
assert(snark.content === 'Hello SnarkTank!', 'snark content stored correctly');
assert(snark.user_id === user.id, 'snark user_id stored correctly');
assert(snark.parent_id === null, 'snark parent_id is null for top-level snark');
assert(snark.created_at !== null, 'snark has created_at timestamp');

const maxContent = 'a'.repeat(280);
const snarkMax = db.prepare('INSERT INTO snarks (user_id, content) VALUES (?, ?)').run(user.id, maxContent);
assert(snarkMax.lastInsertRowid > 0, 'snark with 280 characters can be inserted');

const snarkResult2 = db.prepare('INSERT INTO snarks (user_id, content) VALUES (?, ?)').run(result2.lastInsertRowid, 'Second user snark');
assert(snarkResult2.lastInsertRowid > 0, 'different user can post snark');

const allSnarks = db.prepare('SELECT s.*, u.username, u.display_name FROM snarks s JOIN users u ON s.user_id = u.id ORDER BY s.created_at DESC').all();
assert(allSnarks.length === 3, 'all 3 snarks are retrievable');
assert(allSnarks[0].created_at >= allSnarks[1].created_at, 'snarks ordered by created_at desc');

const userSnarks = db.prepare('SELECT * FROM snarks WHERE user_id = ?').all(user.id);
assert(userSnarks.length === 2, 'can filter snarks by user_id');

const http = require('http');
const app = require('./index');

function request(method, path, body, authToken) {
  return new Promise((resolve, reject) => {
    const options = {
      hostname: 'localhost',
      port: 3002,
      path,
      method,
      headers: { 'Content-Type': 'application/json', 'Connection': 'close' },
    };
    if (authToken) options.headers['Authorization'] = `Bearer ${authToken}`;
    const req = http.request(options, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        try { resolve({ status: res.statusCode, body: JSON.parse(data) }); }
        catch { resolve({ status: res.statusCode, body: data }); }
      });
    });
    req.on('error', reject);
    if (body) req.write(JSON.stringify(body));
    req.end();
  });
}

async function runApiTests() {
  db.exec("DELETE FROM likes; DELETE FROM follows; DELETE FROM snarks; DELETE FROM users;");

  const server = app.listen(3002);
  await new Promise(r => setTimeout(r, 100));

  try {
    const reg = await request('POST', '/api/auth/register', { username: 'timeline_user', displayName: 'Timeline User', password: 'pass123' });
    const loginToken = reg.body.token;

    for (let i = 0; i < 5; i++) {
      await request('POST', '/api/snarks', { content: `Snark number ${i + 1}` }, loginToken);
    }

    const timeline = await request('GET', '/api/snarks');
    assert(timeline.status === 200, 'GET /api/snarks returns 200');
    assert(Array.isArray(timeline.body), 'GET /api/snarks returns array');
    assert(timeline.body.length === 5, 'timeline returns all 5 snarks');
    assert(timeline.body[0].content === 'Snark number 5', 'timeline snarks are newest first');
    assert(timeline.body[4].content === 'Snark number 1', 'oldest snark is last');

    assert(timeline.body[0].author !== undefined, 'snark has author object');
    assert(timeline.body[0].author.displayName === 'Timeline User', 'snark author has displayName');
    assert(timeline.body[0].author.username === 'timeline_user', 'snark author has username');
    assert(timeline.body[0].createdAt !== undefined, 'snark has createdAt timestamp');
    assert(timeline.body[0].likeCount !== undefined, 'snark has likeCount');
    assert(timeline.body[0].replyCount !== undefined, 'snark has replyCount');

    const page1 = await request('GET', '/api/snarks?limit=2&offset=0');
    assert(page1.body.length === 2, 'pagination limit=2 returns 2 snarks');
    assert(page1.body[0].content === 'Snark number 5', 'page 1 starts with newest');

    const page2 = await request('GET', '/api/snarks?limit=2&offset=2');
    assert(page2.body.length === 2, 'pagination offset=2 returns next 2 snarks');
    assert(page2.body[0].content === 'Snark number 3', 'page 2 starts correctly');

    const page3 = await request('GET', '/api/snarks?limit=2&offset=4');
    assert(page3.body.length === 1, 'last page returns remaining snarks');

    const reg2 = await request('POST', '/api/auth/register', { username: 'like_user', displayName: 'Like User', password: 'pass123' });
    const likeToken = reg2.body.token;

    const snarkForLike = await request('POST', '/api/snarks', { content: 'Like this snark' }, loginToken);
    const snarkId = snarkForLike.body.id;

    const likeRes = await request('POST', `/api/snarks/${snarkId}/like`, {}, likeToken);
    assert(likeRes.status === 200, 'POST /api/snarks/:id/like returns 200');
    assert(likeRes.body.liked === true, 'like response has liked=true');
    assert(likeRes.body.likeCount === 1, 'like count is 1 after first like');

    const doubleLike = await request('POST', `/api/snarks/${snarkId}/like`, {}, likeToken);
    assert(doubleLike.status === 409, 'cannot like same snark twice');

    const selfLike = await request('POST', `/api/snarks/${snarkId}/like`, {}, loginToken);
    assert(selfLike.status === 200, 'different user can also like the snark');
    assert(selfLike.body.likeCount === 2, 'like count is 2 after second user likes');

    const unlikeRes = await request('DELETE', `/api/snarks/${snarkId}/like`, {}, likeToken);
    assert(unlikeRes.status === 200, 'DELETE /api/snarks/:id/like returns 200');
    assert(unlikeRes.body.liked === false, 'unlike response has liked=false');
    assert(unlikeRes.body.likeCount === 1, 'like count decreases after unlike');

    const timelineWithLikes = await request('GET', '/api/snarks', null, loginToken);
    const likedSnark = timelineWithLikes.body.find((s) => s.id === snarkId);
    assert(likedSnark.likedByMe === true, 'likedByMe is true for snark liked by current user');
    assert(likedSnark.likeCount === 1, 'timeline shows correct like count');

    const timelineOther = await request('GET', '/api/snarks', null, likeToken);
    const otherView = timelineOther.body.find((s) => s.id === snarkId);
    assert(otherView.likedByMe === false, 'likedByMe is false for unliked user');

    const noAuthTimeline = await request('GET', '/api/snarks');
    const noAuthSnark = noAuthTimeline.body.find((s) => s.id === snarkId);
    assert(noAuthSnark.likedByMe === false, 'likedByMe is false when not authenticated');

    const likeNoAuth = await request('POST', `/api/snarks/${snarkId}/like`, {});
    assert(likeNoAuth.status === 401, 'like requires authentication');

    const likeNotFound = await request('POST', '/api/snarks/99999/like', {}, likeToken);
    assert(likeNotFound.status === 404, 'like returns 404 for non-existent snark');

    const profileRes = await request('GET', '/api/users/timeline_user');
    assert(profileRes.status === 200, 'GET /api/users/:username returns 200');
    assert(profileRes.body.username === 'timeline_user', 'profile has correct username');
    assert(profileRes.body.displayName === 'Timeline User', 'profile has correct displayName');
    assert(profileRes.body.bio !== undefined, 'profile has bio field');
    assert(profileRes.body.createdAt !== undefined, 'profile has createdAt field');
    assert(Array.isArray(profileRes.body.snarks), 'profile has snarks array');
    assert(profileRes.body.snarks.length > 0, 'profile snarks are not empty');
    assert(profileRes.body.snarks[0].author.username === 'timeline_user', 'profile snarks belong to the user');

    const profileNotFound = await request('GET', '/api/users/nonexistent_user_xyz');
    assert(profileNotFound.status === 404, 'GET /api/users/:username returns 404 for unknown user');

    const otherProfile = await request('GET', '/api/users/like_user');
    assert(otherProfile.status === 200, 'can view other user profile');
    assert(otherProfile.body.snarks.length === 0, 'user with no snarks has empty snarks array');

    const updateBio = await request('PUT', '/api/users/profile', { bio: 'I love snarking!' }, loginToken);
    assert(updateBio.status === 200, 'PUT /api/users/profile returns 200');
    assert(updateBio.body.bio === 'I love snarking!', 'bio is updated correctly');

    const profileAfterBio = await request('GET', '/api/users/timeline_user');
    assert(profileAfterBio.body.bio === 'I love snarking!', 'bio persists after update');

    const emptyBio = await request('PUT', '/api/users/profile', { bio: '' }, loginToken);
    assert(emptyBio.status === 200, 'bio can be set to empty string');
    assert(emptyBio.body.bio === '', 'empty bio is returned correctly');

    const longBio = await request('PUT', '/api/users/profile', { bio: 'x'.repeat(161) }, loginToken);
    assert(longBio.status === 400, 'bio over 160 chars is rejected');

    const bioNoAuth = await request('PUT', '/api/users/profile', { bio: 'test' });
    assert(bioNoAuth.status === 401, 'bio update requires authentication');

    const profileWithAuth = await request('GET', '/api/users/timeline_user', null, loginToken);
    assert(profileWithAuth.status === 200, 'profile works with auth token');
    assert(profileWithAuth.body.followerCount !== undefined, 'profile has followerCount');
    assert(profileWithAuth.body.followingCount !== undefined, 'profile has followingCount');
    assert(profileWithAuth.body.followedByMe !== undefined, 'profile has followedByMe');

    const likeUserId = reg2.body.user.id;
    const timelineUserId = reg.body.user.id;

    const followRes = await request('POST', `/api/users/${timelineUserId}/follow`, {}, likeToken);
    assert(followRes.status === 200, 'POST /api/users/:id/follow returns 200');
    assert(followRes.body.following === true, 'follow response has following=true');
    assert(followRes.body.followerCount === 1, 'follower count is 1 after follow');

    const doubleFollow = await request('POST', `/api/users/${timelineUserId}/follow`, {}, likeToken);
    assert(doubleFollow.status === 409, 'cannot follow same user twice');

    const selfFollow = await request('POST', `/api/users/${likeUserId}/follow`, {}, likeToken);
    assert(selfFollow.status === 400, 'cannot follow yourself');

    const profileAfterFollow = await request('GET', '/api/users/timeline_user', null, likeToken);
    assert(profileAfterFollow.body.followerCount === 1, 'profile shows 1 follower');
    assert(profileAfterFollow.body.followedByMe === true, 'followedByMe is true for follower');

    const profileNotFollower = await request('GET', '/api/users/like_user', null, loginToken);
    assert(profileNotFollower.body.followedByMe === false, 'followedByMe is false when not following');
    assert(profileNotFollower.body.followerCount === 0, 'non-followed user has 0 followers');

    const unfollowRes = await request('DELETE', `/api/users/${timelineUserId}/follow`, {}, likeToken);
    assert(unfollowRes.status === 200, 'DELETE /api/users/:id/follow returns 200');
    assert(unfollowRes.body.following === false, 'unfollow response has following=false');
    assert(unfollowRes.body.followerCount === 0, 'follower count is 0 after unfollow');

    const doubleUnfollow = await request('DELETE', `/api/users/${timelineUserId}/follow`, {}, likeToken);
    assert(doubleUnfollow.status === 404, 'cannot unfollow user not being followed');

    const followNoAuth = await request('POST', `/api/users/${timelineUserId}/follow`, {});
    assert(followNoAuth.status === 401, 'follow requires authentication');

    const followNotFound = await request('POST', '/api/users/99999/follow', {}, likeToken);
    assert(followNotFound.status === 404, 'follow returns 404 for non-existent user');

  } finally {
    await new Promise(r => server.close(r));
  }
}

runApiTests().then(() => {
  console.log(`\nResults: ${passed} passed, ${failed} failed`);
  db.exec("DELETE FROM likes; DELETE FROM follows; DELETE FROM snarks; DELETE FROM users;");
  db.close();
  if (failed > 0) process.exit(1);
}).catch(err => {
  console.error('Test error:', err);
  process.exit(1);
});
