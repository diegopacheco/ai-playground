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

console.log(`\nResults: ${passed} passed, ${failed} failed`);

db.exec("DELETE FROM likes; DELETE FROM follows; DELETE FROM snarks; DELETE FROM users;");
db.close();

if (failed > 0) process.exit(1);
