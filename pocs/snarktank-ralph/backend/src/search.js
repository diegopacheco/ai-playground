const express = require('express');
const jwt = require('jsonwebtoken');
const { db } = require('./db');
const { JWT_SECRET } = require('./auth');

const router = express.Router();

router.get('/', (req, res) => {
  const q = (req.query.q || '').trim();
  if (!q) {
    return res.json({ users: [], snarks: [] });
  }

  let authUserId = null;
  const authHeader = req.headers.authorization;
  if (authHeader && authHeader.startsWith('Bearer ')) {
    try {
      const decoded = jwt.verify(authHeader.slice(7), JWT_SECRET);
      authUserId = decoded.id;
    } catch {}
  }

  const searchTerm = `%${q}%`;

  const users = db.prepare(`
    SELECT id, username, display_name, bio, created_at,
           (SELECT COUNT(*) FROM follows WHERE following_id = users.id) as follower_count,
           (SELECT COUNT(*) FROM follows WHERE follower_id = users.id) as following_count,
           (SELECT COUNT(*) FROM follows WHERE follower_id = ? AND following_id = users.id) as followed_by_me
    FROM users
    WHERE username LIKE ? COLLATE NOCASE OR display_name LIKE ? COLLATE NOCASE
    ORDER BY username ASC
    LIMIT 20
  `).all(authUserId || 0, searchTerm, searchTerm);

  const snarks = db.prepare(`
    SELECT s.id, s.content, s.created_at, s.parent_id,
           u.id as user_id, u.username, u.display_name,
           (SELECT COUNT(*) FROM likes WHERE snark_id = s.id) as like_count,
           (SELECT COUNT(*) FROM snarks WHERE parent_id = s.id) as reply_count,
           (SELECT COUNT(*) FROM likes WHERE snark_id = s.id AND user_id = ?) as liked_by_me
    FROM snarks s
    JOIN users u ON s.user_id = u.id
    WHERE s.content LIKE ? COLLATE NOCASE AND s.parent_id IS NULL
    ORDER BY s.created_at DESC, s.id DESC
    LIMIT 20
  `).all(authUserId || 0, searchTerm);

  res.json({
    users: users.map(u => ({
      id: u.id,
      username: u.username,
      displayName: u.display_name,
      bio: u.bio || '',
      createdAt: u.created_at,
      followerCount: u.follower_count,
      followingCount: u.following_count,
      followedByMe: u.followed_by_me > 0,
    })),
    snarks: snarks.map(s => ({
      id: s.id,
      content: s.content,
      createdAt: s.created_at,
      parentId: s.parent_id,
      likeCount: s.like_count,
      replyCount: s.reply_count,
      likedByMe: s.liked_by_me > 0,
      author: {
        id: s.user_id,
        username: s.username,
        displayName: s.display_name,
      },
    })),
  });
});

module.exports = { router };
