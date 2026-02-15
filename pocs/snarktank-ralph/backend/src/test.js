const { db, initializeDatabase } = require('./db');

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

console.log(`\nResults: ${passed} passed, ${failed} failed`);

db.close();

if (failed > 0) process.exit(1);
