const express = require('express');
const jwt = require('jsonwebtoken');
const { db } = require('./db');
const { authMiddleware, JWT_SECRET } = require('./auth');

const router = express.Router();

router.post('/', authMiddleware, (req, res) => {
  const { content, parentId } = req.body;

  if (!content || content.trim().length === 0) {
    return res.status(400).json({ error: 'Snark content cannot be empty' });
  }

  if (content.length > 280) {
    return res.status(400).json({ error: 'Snark cannot exceed 280 characters' });
  }

  if (parentId) {
    const parent = db.prepare('SELECT id FROM snarks WHERE id = ?').get(parentId);
    if (!parent) {
      return res.status(404).json({ error: 'Parent snark not found' });
    }
  }

  const result = db.prepare('INSERT INTO snarks (user_id, content, parent_id) VALUES (?, ?, ?)').run(req.user.id, content.trim(), parentId || null);

  const snark = db.prepare(`
    SELECT s.id, s.content, s.created_at, s.parent_id,
           u.id as user_id, u.username, u.display_name
    FROM snarks s
    JOIN users u ON s.user_id = u.id
    WHERE s.id = ?
  `).get(result.lastInsertRowid);

  res.status(201).json({
    id: snark.id,
    content: snark.content,
    createdAt: snark.created_at,
    parentId: snark.parent_id,
    likeCount: 0,
    replyCount: 0,
    likedByMe: false,
    author: {
      id: snark.user_id,
      username: snark.username,
      displayName: snark.display_name,
    },
  });
});

router.get('/following', authMiddleware, (req, res) => {
  const limit = Math.min(parseInt(req.query.limit) || 20, 50);
  const offset = parseInt(req.query.offset) || 0;

  const snarks = db.prepare(`
    SELECT s.id, s.content, s.created_at, s.parent_id,
           u.id as user_id, u.username, u.display_name,
           (SELECT COUNT(*) FROM likes WHERE snark_id = s.id) as like_count,
           (SELECT COUNT(*) FROM snarks WHERE parent_id = s.id) as reply_count,
           (SELECT COUNT(*) FROM likes WHERE snark_id = s.id AND user_id = ?) as liked_by_me
    FROM snarks s
    JOIN users u ON s.user_id = u.id
    WHERE s.parent_id IS NULL
      AND s.user_id IN (SELECT following_id FROM follows WHERE follower_id = ?)
    ORDER BY s.created_at DESC, s.id DESC
    LIMIT ? OFFSET ?
  `).all(req.user.id, req.user.id, limit, offset);

  res.json(snarks.map(s => ({
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
  })));
});

router.get('/', (req, res) => {
  const limit = Math.min(parseInt(req.query.limit) || 20, 50);
  const offset = parseInt(req.query.offset) || 0;

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
    WHERE s.parent_id IS NULL
    ORDER BY s.created_at DESC, s.id DESC
    LIMIT ? OFFSET ?
  `).all(authUserId || 0, limit, offset);

  res.json(snarks.map(s => ({
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
  })));
});

router.get('/:id', (req, res) => {
  const snarkId = parseInt(req.params.id);

  let authUserId = null;
  const authHeader = req.headers.authorization;
  if (authHeader && authHeader.startsWith('Bearer ')) {
    try {
      const decoded = jwt.verify(authHeader.slice(7), JWT_SECRET);
      authUserId = decoded.id;
    } catch {}
  }

  const snark = db.prepare(`
    SELECT s.id, s.content, s.created_at, s.parent_id,
           u.id as user_id, u.username, u.display_name,
           (SELECT COUNT(*) FROM likes WHERE snark_id = s.id) as like_count,
           (SELECT COUNT(*) FROM snarks WHERE parent_id = s.id) as reply_count,
           (SELECT COUNT(*) FROM likes WHERE snark_id = s.id AND user_id = ?) as liked_by_me
    FROM snarks s
    JOIN users u ON s.user_id = u.id
    WHERE s.id = ?
  `).get(authUserId || 0, snarkId);

  if (!snark) {
    return res.status(404).json({ error: 'Snark not found' });
  }

  const replies = db.prepare(`
    SELECT s.id, s.content, s.created_at, s.parent_id,
           u.id as user_id, u.username, u.display_name,
           (SELECT COUNT(*) FROM likes WHERE snark_id = s.id) as like_count,
           (SELECT COUNT(*) FROM snarks WHERE parent_id = s.id) as reply_count,
           (SELECT COUNT(*) FROM likes WHERE snark_id = s.id AND user_id = ?) as liked_by_me
    FROM snarks s
    JOIN users u ON s.user_id = u.id
    WHERE s.parent_id = ?
    ORDER BY s.created_at ASC, s.id ASC
  `).all(authUserId || 0, snarkId);

  const formatSnark = (s) => ({
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
  });

  res.json({
    ...formatSnark(snark),
    replies: replies.map(formatSnark),
  });
});

router.post('/:id/like', authMiddleware, (req, res) => {
  const snarkId = parseInt(req.params.id);
  const snark = db.prepare('SELECT id FROM snarks WHERE id = ?').get(snarkId);
  if (!snark) {
    return res.status(404).json({ error: 'Snark not found' });
  }

  const existing = db.prepare('SELECT * FROM likes WHERE user_id = ? AND snark_id = ?').get(req.user.id, snarkId);
  if (existing) {
    return res.status(409).json({ error: 'Already liked' });
  }

  db.prepare('INSERT INTO likes (user_id, snark_id) VALUES (?, ?)').run(req.user.id, snarkId);
  const likeCount = db.prepare('SELECT COUNT(*) as count FROM likes WHERE snark_id = ?').get(snarkId).count;
  res.json({ liked: true, likeCount });
});

router.delete('/:id/like', authMiddleware, (req, res) => {
  const snarkId = parseInt(req.params.id);
  const result = db.prepare('DELETE FROM likes WHERE user_id = ? AND snark_id = ?').run(req.user.id, snarkId);
  const likeCount = db.prepare('SELECT COUNT(*) as count FROM likes WHERE snark_id = ?').get(snarkId).count;
  res.json({ liked: false, likeCount });
});

module.exports = { router };
