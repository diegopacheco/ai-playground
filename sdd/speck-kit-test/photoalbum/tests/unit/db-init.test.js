import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { initializeDatabase, getDatabase, closeDatabase, exportDatabase, importDatabase } from '../../src/db/init.js';

describe('Database Initialization', () => {
  afterEach(() => {
    closeDatabase();
  });

  it('should initialize database successfully', async () => {
    const db = await initializeDatabase();
    expect(db).toBeDefined();
    expect(db).toBeTruthy();
  });

  it('should return same database instance on multiple calls', async () => {
    const db1 = await initializeDatabase();
    const db2 = await initializeDatabase();
    expect(db1).toBe(db2);
  });

  it('should allow getting database after initialization', async () => {
    await initializeDatabase();
    const db = getDatabase();
    expect(db).toBeDefined();
  });

  it('should throw error when getting database before initialization', () => {
    expect(() => getDatabase()).toThrow('Database not initialized');
  });

  it('should export database as binary array', async () => {
    await initializeDatabase();
    const exported = exportDatabase();
    expect(exported).toBeInstanceOf(Uint8Array);
    expect(exported.length).toBeGreaterThan(0);
  });

  it('should import database from binary array', async () => {
    const db1 = await initializeDatabase();
    const exported = exportDatabase();
    closeDatabase();

    const db2 = importDatabase(exported);
    expect(db2).toBeDefined();
  });

  it('should close database properly', async () => {
    await initializeDatabase();
    closeDatabase();
    expect(() => getDatabase()).toThrow('Database not initialized');
  });
});
