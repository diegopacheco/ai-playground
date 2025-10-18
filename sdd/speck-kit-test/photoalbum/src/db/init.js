import initSqlJs from 'sql.js';

let SQL = null;
let db = null;

export async function initializeDatabase() {
  if (db) {
    return db;
  }

  try {
    const isNode = typeof process !== 'undefined' && process.versions && process.versions.node;

    if (isNode) {
      SQL = await initSqlJs();
    } else {
      SQL = await initSqlJs({
        locateFile: file => `https://sql.js.org/dist/${file}`
      });
    }

    db = new SQL.Database();

    db.run('PRAGMA foreign_keys = ON');

    console.log('SQLite database initialized in memory');
    return db;
  } catch (error) {
    console.error('Failed to initialize database:', error);
    throw new Error(`Database initialization failed: ${error.message}`);
  }
}

export function getDatabase() {
  if (!db) {
    throw new Error('Database not initialized. Call initializeDatabase() first.');
  }
  return db;
}

export function closeDatabase() {
  if (db) {
    db.close();
    db = null;
    SQL = null;
    console.log('Database closed');
  }
}

export function exportDatabase() {
  if (!db) {
    throw new Error('Database not initialized');
  }
  return db.export();
}

export async function importDatabase(dataArray) {
  if (!SQL) {
    const isNode = typeof process !== 'undefined' && process.versions && process.versions.node;

    if (isNode) {
      SQL = await initSqlJs();
    } else {
      SQL = await initSqlJs({
        locateFile: file => `https://sql.js.org/dist/${file}`
      });
    }
  }

  if (db) {
    db.close();
  }

  db = new SQL.Database(dataArray);
  db.run('PRAGMA foreign_keys = ON');
  console.log('Database imported from binary data');
  return db;
}
