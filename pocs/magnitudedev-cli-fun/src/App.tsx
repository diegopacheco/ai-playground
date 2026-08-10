/**
 * TanStack Table V9 Application
 * A clean, modular, light-themed data table application
 * 
 * Design Philosophy:
 * - Refined minimalism with sophisticated typography
 * - Generous whitespace and careful spacing
 * - Subtle shadows and borders for depth
 * - Clean, professional aesthetic
 */

import { useState } from 'react';
import { Routes, Route, Link, useLocation } from 'react-router-dom';
import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  useReactTable,
  getPaginationRowModel,
  getSortedRowModel,
  SortingState,
} from '@tanstack/react-table';
import { format } from 'date-fns';

// Custom hook for sorting state
function useSorting() {
  const [sorting, setSorting] = useState<SortingState>([]);

  return {
    sorting,
    setSorting,
  };
}

// Column helper
const columnHelper = createColumnHelper<any>();

// Sample data
const sampleData = [
  { id: 1, name: 'Alice Johnson', email: 'alice@example.com', role: 'Administrator', status: 'Active', joined: new Date('2024-01-15') },
  { id: 2, name: 'Bob Smith', email: 'bob@example.com', role: 'Editor', status: 'Active', joined: new Date('2024-02-20') },
  { id: 3, name: 'Carol Williams', email: 'carol@example.com', role: 'Viewer', status: 'Inactive', joined: new Date('2024-03-10') },
  { id: 4, name: 'David Brown', email: 'david@example.com', role: 'Editor', status: 'Active', joined: new Date('2024-04-05') },
  { id: 5, name: 'Eva Martinez', email: 'eva@example.com', role: 'Administrator', status: 'Active', joined: new Date('2024-05-12') },
  { id: 6, name: 'Frank Garcia', email: 'frank@example.com', role: 'Viewer', status: 'Pending', joined: new Date('2024-06-18') },
  { id: 7, name: 'Grace Lee', email: 'grace@example.com', role: 'Editor', status: 'Active', joined: new Date('2024-07-22') },
  { id: 8, name: 'Henry Wilson', email: 'henry@example.com', role: 'Viewer', status: 'Inactive', joined: new Date('2024-08-01') },
];

// Main App component
function App() {
  const location = useLocation();
  const isDashboard = location.pathname === '/';

  return (
    <div className="app">
      <div className={`sidebar ${isDashboard ? 'collapsed' : ''}`}>
        <div className="sidebar-header">
          <div className="logo">
            <span className="logo-icon">⌘</span>
            <span className="logo-text">Magnitude</span>
          </div>
          <button className="sidebar-toggle" onClick={() => {}}>
            ☰
          </button>
        </div>
        <nav className="sidebar-nav">
          <Link to="/" className="nav-item active">
            <span className="nav-icon">📊</span>
            <span className="nav-text">Dashboard</span>
          </Link>
          <Link to="/users" className="nav-item">
            <span className="nav-icon">👥</span>
            <span className="nav-text">Users</span>
          </Link>
          <Link to="/analytics" className="nav-item">
            <span className="nav-icon">📈</span>
            <span className="nav-text">Analytics</span>
          </Link>
          <Link to="/settings" className="nav-item">
            <span className="nav-icon">⚙️</span>
            <span className="nav-text">Settings</span>
          </Link>
        </nav>
        <div className="sidebar-footer">
          <div className="version">v1.0.0</div>
        </div>
      </div>
      <main className="main-content">
        <header className="main-header">
          <h1 className="page-title">Dashboard</h1>
          <div className="header-actions">
            <div className="user-profile">
              <span className="user-avatar">DJ</span>
              <span className="user-name">Diego Pacheco</span>
            </div>
          </div>
        </header>
        <div className="page-content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/users" element={<UsersTable />} />
            <Route path="/analytics" element={<Analytics />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </div>
      </main>
    </div>
  );
}

// Dashboard component
function Dashboard() {
  return (
    <div className="dashboard">
      <section className="stats-grid">
        <div className="stat-card">
          <div className="stat-icon">👥</div>
          <div className="stat-content">
            <div className="stat-value">1,234</div>
            <div className="stat-label">Total Users</div>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon">📊</div>
          <div className="stat-content">
            <div className="stat-value">+12%</div>
            <div className="stat-label">Growth</div>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon">⚡</div>
          <div className="stat-content">
            <div className="stat-value">99.9%</div>
            <div className="stat-label">Uptime</div>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon">🔄</div>
          <div className="stat-content">
            <div className="stat-value">456</div>
            <div className="stat-label">Active Sessions</div>
          </div>
        </div>
      </section>
      <section className="data-section">
        <h2 className="section-title">Recent Activity</h2>
        <div className="activity-list">
          <ActivityItem time="2m ago" description="New user registered" />
          <ActivityItem time="15m ago" description="System backup completed" />
          <ActivityItem time="1h ago" description="Settings updated" />
          <ActivityItem time="3h ago" description="Report generated" />
        </div>
      </section>
    </div>
  );
}

// Activity item component
function ActivityItem({ time, description }: { time: string; description: string }) {
  return (
    <div className="activity-item">
      <div className="activity-time">{time}</div>
      <div className="activity-description">{description}</div>
    </div>
  );
}

// Users Table component with TanStack Table V9
function UsersTable() {
  const { sorting, setSorting } = useSorting();

  const columns = [
    columnHelper.accessor('name', {
      header: 'Name',
      cell: (info) => (
        <div className="cell-name">{info.getValue()}</div>
      ),
    }),
    columnHelper.accessor('email', {
      header: 'Email',
      cell: (info) => (
        <a href={`mailto:${info.getValue()}`} className="cell-email">
          {info.getValue()}
        </a>
      ),
    }),
    columnHelper.accessor('role', {
      header: 'Role',
      cell: (info) => (
        <span className={`role-badge role-${info.getValue().toLowerCase()}`}>
          {info.getValue()}
        </span>
      ),
    }),
    columnHelper.accessor('status', {
      header: 'Status',
      cell: (info) => (
        <span className={`status-badge status-${info.getValue().toLowerCase()}`}>
          {info.getValue()}
        </span>
      ),
    }),
    columnHelper.accessor('joined', {
      header: 'Joined',
      cell: (info) => (
        <div className="cell-date">{format(info.getValue(), 'MMM d, yyyy')}</div>
      ),
    }),
    columnHelper.display({
      id: 'actions',
      header: 'Actions',
      cell: () => (
        <div className="cell-actions">
          <button className="action-btn" aria-label="Edit">✏️</button>
          <button className="action-btn" aria-label="Delete">🗑️</button>
        </div>
      ),
    }),
  ];

  const table = useReactTable({
    data: sampleData,
    columns,
    getCoreRowModel: getCoreRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    getSortedRowModel: getSortedRowModel(),
    onSortingChange: setSorting,
    state: {
      sorting,
    },
  });

  return (
    <div className="table-container">
      <div className="table-toolbar">
        <div className="search-box">
          <span className="search-icon">🔍</span>
          <input
            type="text"
            placeholder="Search users..."
            className="search-input"
          />
        </div>
        <div className="table-actions">
          <button className="btn btn-primary">
            <span className="btn-icon">+</span>
            Add User
          </button>
        </div>
      </div>
      <div className="table-wrapper">
        <table className="data-table">
          <thead className="table-header">
            {table.getHeaderGroups().map((headerGroup) => (
              <tr key={headerGroup.id} className="table-row">
                {headerGroup.headers.map((header) => {
                  return (
                    <th
                      key={header.id}
                      className={header.column.getCanSort() ? 'table-head sortable' : 'table-head'}
                      onClick={() => header.column.getToggleSortingHandler()}
                    >
                      <div className="table-head-content">
                        {flexRender(header.column.columnDef.header, header.getContext())}
                        {header.column.getCanSort() && (
                          <span className="sort-indicator">
                            {header.column.getIsSorted() === 'asc' ? '↑' : '↓'}
                          </span>
                        )}
                      </div>
                    </th>
                  );
                })}
              </tr>
            ))}
          </thead>
          <tbody className="table-body">
            {table.getRowModel().rows.map((row) => (
              <tr key={row.id} className="table-row">
                {row.getVisibleCells().map((cell) => (
                  <td key={cell.id} className="table-cell">
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="table-pagination">
        <div className="pagination-info">
          Showing {table.getRowModel().rows.length} of {sampleData.length} entries
        </div>
        <div className="pagination-controls">
          <button
            className="pagination-btn"
            onClick={() => table.previousPage()}
            disabled={!table.getCanPreviousPage()}
          >
            ← Previous
          </button>
          <button
            className="pagination-btn"
            onClick={() => table.nextPage()}
            disabled={!table.getCanNextPage()}
          >
            Next →
          </button>
        </div>
      </div>
    </div>
  );
}

// Analytics placeholder
function Analytics() {
  return (
    <div className="analytics-placeholder">
      <div className="placeholder-icon">📈</div>
      <h2>Analytics Dashboard</h2>
      <p>Analytics features coming soon...</p>
    </div>
  );
}

// Settings placeholder
function Settings() {
  return (
    <div className="settings-placeholder">
      <div className="placeholder-icon">⚙️</div>
      <h2>Settings</h2>
      <p>Configuration options coming soon...</p>
    </div>
  );
}

// CSS styles
const styles = `
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

:root {
  --color-background: #ffffff;
  --color-surface: #f8f9fa;
  --color-primary: #2563eb;
  --color-primary-hover: #1d4ed8;
  --color-text-primary: #1f2937;
  --color-text-secondary: #6b7280;
  --color-border: #e5e7eb;
  --color-border-light: #f3f4f6;
  --color-success: #10b981;
  --color-warning: #f59e0b;
  --color-error: #ef4444;
  --color-info: #3b82f6;
  --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
  --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
  --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -4px rgba(0, 0, 0, 0.1);
  --radius-sm: 0.25rem;
  --radius-md: 0.5rem;
  --radius-lg: 0.75rem;
  --font-sans: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
  --font-mono: 'SF Mono', 'Fira Code', 'Fira Mono', Menlo, monospace;
}

.app {
  display: flex;
  min-height: 100vh;
  background: var(--color-background);
  font-family: var(--font-sans);
  color: var(--color-text-primary);
}

.sidebar {
  width: 260px;
  background: var(--color-background);
  border-right: 1px solid var(--color-border);
  display: flex;
  flex-direction: column;
  transition: width 0.3s ease;
}

.sidebar-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1.5rem 1.5rem 1rem 1.5rem;
  border-bottom: 1px solid var(--color-border);
}

.logo {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--color-text-primary);
}

.logo-icon {
  font-size: 1.75rem;
}

.sidebar-toggle {
  background: none;
  border: none;
  cursor: pointer;
  font-size: 1.25rem;
  color: var(--color-text-secondary);
}

.sidebar-nav {
  flex: 1;
  padding: 1rem 0;
  overflow-y: auto;
}

.nav-item {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.875rem 1.5rem;
  color: var(--color-text-secondary);
  text-decoration: none;
  transition: all 0.2s ease;
  border-left: 3px solid transparent;
}

.nav-item:hover {
  background: var(--color-surface);
  color: var(--color-text-primary);
}

.nav-item.active {
  background: var(--color-primary);
  color: white;
  border-left-color: white;
}

.nav-icon {
  font-size: 1.25rem;
}

.nav-text {
  font-weight: 500;
}

.sidebar-footer {
  padding: 1rem 1.5rem;
  border-top: 1px solid var(--color-border);
}

.version {
  color: var(--color-text-secondary);
  font-size: 0.875rem;
}

.main-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.main-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1.25rem 2rem;
  border-bottom: 1px solid var(--color-border);
  background: var(--color-background);
}

.page-title {
  font-size: 1.75rem;
  font-weight: 600;
  color: var(--color-text-primary);
}

.header-actions {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.user-profile {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.user-avatar {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background: var(--color-primary);
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 600;
  font-size: 0.875rem;
}

.user-name {
  font-weight: 500;
  color: var(--color-text-primary);
}

.page-content {
  flex: 1;
  padding: 2rem;
  overflow-y: auto;
}

.dashboard {
  display: grid;
  gap: 2rem;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 1.5rem;
}

.stat-card {
  background: var(--color-background);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  padding: 1.5rem;
  display: flex;
  align-items: center;
  gap: 1rem;
  transition: all 0.2s ease;
}

.stat-card:hover {
  box-shadow: var(--shadow-md);
  border-color: var(--color-primary);
}

.stat-icon {
  font-size: 2.5rem;
  opacity: 0.5;
}

.stat-content {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.stat-value {
  font-size: 2rem;
  font-weight: 700;
  color: var(--color-text-primary);
}

.stat-label {
  font-size: 0.875rem;
  color: var(--color-text-secondary);
}

.data-section {
  background: var(--color-background);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  padding: 1.5rem;
}

.section-title {
  font-size: 1.25rem;
  font-weight: 600;
  margin-bottom: 1.5rem;
  color: var(--color-text-primary);
}

.activity-list {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.activity-item {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 0.75rem;
  border-radius: var(--radius-md);
  transition: background 0.2s ease;
}

.activity-item:hover {
  background: var(--color-surface);
}

.activity-time {
  font-size: 0.875rem;
  color: var(--color-text-secondary);
  min-width: 80px;
}

.activity-description {
  font-size: 0.875rem;
  color: var(--color-text-primary);
}

.table-container {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.table-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
}

.search-box {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  padding: 0.5rem 0.75rem;
  flex: 1;
  max-width: 400px;
}

.search-icon {
  font-size: 1rem;
  color: var(--color-text-secondary);
}

.search-input {
  flex: 1;
  border: none;
  background: transparent;
  font-size: 0.875rem;
  color: var(--color-text-primary);
  outline: none;
}

.search-input::placeholder {
  color: var(--color-text-secondary);
}

.table-actions {
  display: flex;
  gap: 0.5rem;
}

.btn {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.625rem 1rem;
  border-radius: var(--radius-md);
  font-size: 0.875rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
  border: 1px solid var(--color-border);
}

.btn-primary {
  background: var(--color-primary);
  color: white;
  border-color: var(--color-primary);
}

.btn-primary:hover {
  background: var(--color-primary-hover);
}

.table-wrapper {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  overflow: hidden;
}

.data-table {
  width: 100%;
  border-collapse: collapse;
}

.table-header {
  background: var(--color-background);
  border-bottom: 1px solid var(--color-border);
}

.table-row {
  transition: background 0.15s ease;
}

.table-row:hover {
  background: var(--color-surface);
}

.table-head {
  text-align: left;
  padding: 1rem 1.5rem;
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--color-text-secondary);
  border-bottom: 1px solid var(--color-border);
}

.table-head.sortable {
  cursor: pointer;
}

.table-head-content {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.sort-indicator {
  color: var(--color-text-secondary);
  font-size: 0.7rem;
}

.table-cell {
  padding: 1rem 1.5rem;
  font-size: 0.875rem;
  color: var(--color-text-primary);
  border-bottom: 1px solid var(--color-border-light);
}

.cell-name {
  font-weight: 500;
}

.cell-email {
  color: var(--color-primary);
  text-decoration: none;
}

.cell-email:hover {
  text-decoration: underline;
}

.cell-date {
  color: var(--color-text-secondary);
  font-size: 0.8rem;
}

.cell-actions {
  display: flex;
  gap: 0.5rem;
}

.action-btn {
  background: none;
  border: none;
  cursor: pointer;
  font-size: 1rem;
  opacity: 0.5;
  transition: opacity 0.2s ease;
  padding: 0.25rem;
}

.action-btn:hover {
  opacity: 1;
}

.status-badge {
  padding: 0.25rem 0.75rem;
  border-radius: 1rem;
  font-size: 0.75rem;
  font-weight: 500;
  text-transform: capitalize;
}

.status-active {
  background: #d1fae5;
  color: #065f46;
}

.status-inactive {
  background: #f3f4f6;
  color: #6b7280;
}

.status-pending {
  background: #fef3c7;
  color: #92400e;
}

.role-badge {
  padding: 0.25rem 0.75rem;
  border-radius: 1rem;
  font-size: 0.75rem;
  font-weight: 500;
  text-transform: capitalize;
  background: var(--color-surface);
  border: 1px solid var(--color-border);
}

.table-pagination {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1rem 1.5rem;
  background: var(--color-background);
  border-top: 1px solid var(--color-border);
}

.pagination-info {
  font-size: 0.875rem;
  color: var(--color-text-secondary);
}

.pagination-controls {
  display: flex;
  gap: 0.5rem;
}

.pagination-btn {
  padding: 0.5rem 1rem;
  border: 1px solid var(--color-border);
  background: var(--color-background);
  border-radius: var(--radius-md);
  font-size: 0.875rem;
  cursor: pointer;
  transition: all 0.2s ease;
}

.pagination-btn:hover:not(:disabled) {
  background: var(--color-surface);
  border-color: var(--color-primary);
}

.pagination-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.analytics-placeholder,
.settings-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 400px;
  text-align: center;
  color: var(--color-text-secondary);
}

.placeholder-icon {
  font-size: 4rem;
  margin-bottom: 1rem;
}

@media (max-width: 768px) {
  .sidebar {
    position: fixed;
    left: 0;
    top: 0;
    height: 100vh;
    z-index: 1000;
  }

  .sidebar.collapsed {
    transform: translateX(-100%);
  }

  .main-header {
    padding: 1rem;
  }

  .page-title {
    font-size: 1.5rem;
  }

  .page-content {
    padding: 1rem;
  }

  .stats-grid {
    grid-template-columns: 1fr;
  }

  .table-toolbar {
    flex-direction: column;
    align-items: flex-start;
  }

  .search-box {
    width: 100%;
    max-width: none;
  }
}
`;

// Inject styles
const styleSheet = document.createElement('style');
styleSheet.textContent = styles;
document.head.appendChild(styleSheet);

export default App;
