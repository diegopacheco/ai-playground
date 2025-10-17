import { initializeDatabase, getDatabase, exportDatabase, importDatabase, closeDatabase } from './init.js';
import { loadSchema, verifySchema } from './schema.js';
import { saveDatabaseToIndexedDB, loadDatabaseFromIndexedDB, clearDatabaseFromIndexedDB } from './persistence.js';

let isInitialized = false;
let saveTimeout = null;

export async function initApp() {
  try {
    console.log('Initializing application database...');

    const existingData = await loadDatabaseFromIndexedDB();

    if (existingData) {
      console.log('Loading existing database from IndexedDB');
      await importDatabase(existingData);
      verifySchema();
    } else {
      console.log('Creating new database');
      await initializeDatabase();
      await loadSchema();
    }

    isInitialized = true;
    console.log('Application database ready');

    return true;
  } catch (error) {
    console.error('Failed to initialize application:', error);
    throw new Error(`Application initialization failed: ${error.message}`);
  }
}

export function executeQuery(sql, params = []) {
  if (!isInitialized) {
    throw new Error('Database not initialized. Call initApp() first.');
  }

  try {
    const db = getDatabase();
    const stmt = db.prepare(sql);
    stmt.bind(params);

    const result = {
      columns: stmt.getColumnNames(),
      values: []
    };

    while (stmt.step()) {
      result.values.push(stmt.get());
    }

    stmt.free();
    return result;
  } catch (error) {
    console.error('Query execution failed:', error, { sql, params });
    throw new Error(`Query failed: ${error.message}`);
  }
}

export function executeUpdate(sql, params = []) {
  if (!isInitialized) {
    throw new Error('Database not initialized. Call initApp() first.');
  }

  try {
    const db = getDatabase();
    db.run(sql, params);

    const changes = db.exec('SELECT changes() as changes')[0].values[0][0];

    debouncedSave();

    return changes;
  } catch (error) {
    console.error('Update execution failed:', error, { sql, params });
    throw new Error(`Update failed: ${error.message}`);
  }
}

export function executeInsert(sql, params = []) {
  if (!isInitialized) {
    throw new Error('Database not initialized. Call initApp() first.');
  }

  try {
    const db = getDatabase();
    db.run(sql, params);

    const lastId = db.exec('SELECT last_insert_rowid() as id')[0].values[0][0];

    debouncedSave();

    return lastId;
  } catch (error) {
    console.error('Insert execution failed:', error, { sql, params });
    throw new Error(`Insert failed: ${error.message}`);
  }
}

export function beginTransaction() {
  const db = getDatabase();
  db.run('BEGIN TRANSACTION');
}

export function commitTransaction() {
  const db = getDatabase();
  db.run('COMMIT');
  debouncedSave();
}

export function rollbackTransaction() {
  const db = getDatabase();
  db.run('ROLLBACK');
}

function debouncedSave() {
  if (saveTimeout) {
    clearTimeout(saveTimeout);
  }

  saveTimeout = setTimeout(async () => {
    try {
      await saveDatabase();
    } catch (error) {
      console.error('Auto-save failed:', error);
    }
  }, 1000);
}

export async function saveDatabase() {
  if (!isInitialized) {
    return;
  }

  try {
    const data = exportDatabase();
    await saveDatabaseToIndexedDB(data);
    console.log('Database saved to IndexedDB');
  } catch (error) {
    console.error('Failed to save database:', error);
    throw error;
  }
}

export async function clearAllData() {
  try {
    await clearDatabaseFromIndexedDB();
    closeDatabase();
    isInitialized = false;
    console.log('All data cleared');
  } catch (error) {
    console.error('Failed to clear data:', error);
    throw error;
  }
}

export function getStats() {
  if (!isInitialized) {
    return null;
  }

  try {
    const albumCount = executeQuery('SELECT COUNT(*) as count FROM Album');
    const photoCount = executeQuery('SELECT COUNT(*) as count FROM Photo');
    const relationCount = executeQuery('SELECT COUNT(*) as count FROM AlbumPhoto');

    return {
      albums: albumCount.values[0][0],
      photos: photoCount.values[0][0],
      relations: relationCount.values[0][0]
    };
  } catch (error) {
    console.error('Failed to get stats:', error);
    return null;
  }
}

export { isInitialized };
