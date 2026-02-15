const express = require('express');
const cors = require('cors');
const { initializeDatabase } = require('./db');
const { router: authRouter } = require('./auth');
const { router: snarksRouter } = require('./snarks');

const app = express();
const PORT = 3001;

app.use(cors());
app.use(express.json());

initializeDatabase();

app.use('/api/auth', authRouter);
app.use('/api/snarks', snarksRouter);

app.get('/api/health', (req, res) => {
  res.json({ status: 'ok' });
});

if (require.main === module) {
  app.listen(PORT, () => {
    console.log(`SnarkTank backend running on port ${PORT}`);
  });
}

module.exports = app;
