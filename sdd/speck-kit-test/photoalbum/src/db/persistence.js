const DB_NAME = 'PhotoAlbumDB';
const DB_VERSION = 1;
const STORE_NAME = 'database';
const DB_KEY = 'sqliteData';

let indexedDB = null;

function openIndexedDB() {
  return new Promise((resolve, reject) => {
    if (typeof window === 'undefined' || !window.indexedDB) {
      resolve(null);
      return;
    }

    const request = window.indexedDB.open(DB_NAME, DB_VERSION);

    request.onerror = () => {
      reject(new Error('Failed to open IndexedDB'));
    };

    request.onsuccess = (event) => {
      indexedDB = event.target.result;
      resolve(indexedDB);
    };

    request.onupgradeneeded = (event) => {
      const db = event.target.result;

      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME);
        console.log('IndexedDB object store created');
      }
    };
  });
}

export async function saveDatabaseToIndexedDB(dataArray) {
  try {
    const db = indexedDB || await openIndexedDB();

    if (!db) {
      console.log('IndexedDB not available, skipping save');
      return;
    }

    return new Promise((resolve, reject) => {
      const transaction = db.transaction([STORE_NAME], 'readwrite');
      const store = transaction.objectStore(STORE_NAME);
      const request = store.put(dataArray, DB_KEY);

      request.onsuccess = () => {
        console.log('Database saved to IndexedDB');
        resolve();
      };

      request.onerror = () => {
        reject(new Error('Failed to save database to IndexedDB'));
      };
    });
  } catch (error) {
    console.error('Error saving to IndexedDB:', error);
    throw error;
  }
}

export async function loadDatabaseFromIndexedDB() {
  try {
    const db = indexedDB || await openIndexedDB();

    if (!db) {
      console.log('IndexedDB not available, skipping load');
      return null;
    }

    return new Promise((resolve, reject) => {
      const transaction = db.transaction([STORE_NAME], 'readonly');
      const store = transaction.objectStore(STORE_NAME);
      const request = store.get(DB_KEY);

      request.onsuccess = () => {
        if (request.result) {
          console.log('Database loaded from IndexedDB');
          resolve(request.result);
        } else {
          console.log('No existing database found in IndexedDB');
          resolve(null);
        }
      };

      request.onerror = () => {
        reject(new Error('Failed to load database from IndexedDB'));
      };
    });
  } catch (error) {
    console.error('Error loading from IndexedDB:', error);
    throw error;
  }
}

export async function clearDatabaseFromIndexedDB() {
  try {
    const db = indexedDB || await openIndexedDB();

    if (!db) {
      console.log('IndexedDB not available, skipping clear');
      return;
    }

    return new Promise((resolve, reject) => {
      const transaction = db.transaction([STORE_NAME], 'readwrite');
      const store = transaction.objectStore(STORE_NAME);
      const request = store.delete(DB_KEY);

      request.onsuccess = () => {
        console.log('Database cleared from IndexedDB');
        resolve();
      };

      request.onerror = () => {
        reject(new Error('Failed to clear database from IndexedDB'));
      };
    });
  } catch (error) {
    console.error('Error clearing IndexedDB:', error);
    throw error;
  }
}

export function closeIndexedDB() {
  if (indexedDB) {
    indexedDB.close();
    indexedDB = null;
    console.log('IndexedDB connection closed');
  }
}
