const express = require('express');
const jwt = require('jsonwebtoken');
const { db } = require('./db');
const { authMiddleware, JWT_SECRET } = require('./auth');

const router = express.Router();

router.put('/profile', authMiddleware, (req, res) => {
  const { bio } = req.body;
  if (bio === undefined || bio === null) {
    return res.status(400).json({ error: 'Bio is required' });
  }
  if (bio.length > 160) {
    return res.status(400).json({ error: 'Bio cannot exceed 160 characters' });
  }
  db.prepare('UPDATE users SET bio = ? WHERE id = ?').run(bio, req.user.id);
  const user = db.prepare('SELECT id, username, display_name, bio, created_at FROM users WHERE id = ?').get(req.user.id);
  res.json({
    id: user.id,
    username: user.username,
    displayName: user.display_name,
    bio: user.bio || '',
    createdAt: user.created_at,
  });
});

router.get('/:username', (req, res) => {
  const user = db.prepare('SELECT id, username, display_name, bio, created_at FROM users WHERE username = ?').get(req.params.username);
  if (!user) {
    return res.status(404).json({ error: 'User not found' });
  }

  let authUserId = null;
  const authHeader = req.headers.authorization;
  if (authHeader && authHeader.startsWith('Bearer ')) {
    try {
      const decoded = jwt.verify(authHeader.slice(7), JWT_SECRET);
      authUserId = decoded.id;
    } catch {}
  }

  const snarks = db.prepare(`
    SELECT s.id, s.content, s.created_at, s.parent_id,
           u.id as user_id, u.username, u.display_name,
           (SELECT COUNT(*) FROM likes WHERE snark_id = s.id) as like_count,
           (SELECT COUNT(*) FROM snarks WHERE parent_id = s.id) as reply_count,
           (SELECT COUNT(*) FROM likes WHERE snark_id = s.id AND user_id = ?) as liked_by_me
    FROM snarks s
    JOIN users u ON s.user_id = u.id
    WHERE s.user_id = ? AND s.parent_id IS NULL
    ORDER BY s.created_at DESC, s.id DESC
  `).all(authUserId || 0, user.id);

  res.json({
    id: user.id,
    username: user.username,
    displayName: user.display_name,
    bio: user.bio || '',
    createdAt: user.created_at,
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
