const express = require('express');
const { db } = require('./db');
const { authMiddleware } = require('./auth');

const router = express.Router();

router.post('/', authMiddleware, (req, res) => {
  const { content } = req.body;

  if (!content || content.trim().length === 0) {
    return res.status(400).json({ error: 'Snark content cannot be empty' });
  }

  if (content.length > 280) {
    return res.status(400).json({ error: 'Snark cannot exceed 280 characters' });
  }

  const result = db.prepare('INSERT INTO snarks (user_id, content) VALUES (?, ?)').run(req.user.id, content.trim());

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
    author: {
      id: snark.user_id,
      username: snark.username,
      displayName: snark.display_name,
    },
  });
});

router.get('/', (req, res) => {
  const limit = Math.min(parseInt(req.query.limit) || 20, 50);
  const offset = parseInt(req.query.offset) || 0;

  const snarks = db.prepare(`
    SELECT s.id, s.content, s.created_at, s.parent_id,
           u.id as user_id, u.username, u.display_name,
           (SELECT COUNT(*) FROM likes WHERE snark_id = s.id) as like_count,
           (SELECT COUNT(*) FROM snarks WHERE parent_id = s.id) as reply_count
    FROM snarks s
    JOIN users u ON s.user_id = u.id
    WHERE s.parent_id IS NULL
    ORDER BY s.created_at DESC
    LIMIT ? OFFSET ?
  `).all(limit, offset);

  res.json(snarks.map(s => ({
    id: s.id,
    content: s.content,
    createdAt: s.created_at,
    parentId: s.parent_id,
    likeCount: s.like_count,
    replyCount: s.reply_count,
    author: {
      id: s.user_id,
      username: s.username,
      displayName: s.display_name,
    },
  })));
});

module.exports = { router };
