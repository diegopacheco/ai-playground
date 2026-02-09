# Changelog

## [0.2.0] - 2026-02-09

### Added
- Post view counter feature to track person record views
- New `post_views` table in SQLite database with foreign key cascade delete
- Backend endpoint `GET /admin/views` returning all persons with view counts
- Backend endpoint `GET /admin/views/{id}` returning view count for specific person
- Admin Panel tab in React frontend displaying view counts
- Automatic view count increment on `GET /persons/{id}` requests
- Navigation between Person CRUD and Admin Panel views
- Refresh button in Admin Panel to reload view counts

### Changed
- Backend `GET /persons/{id}` now increments view count via UPSERT
- Backend `POST /persons` initializes post_views entry with view_count=0
- Frontend App.js updated with tabbed navigation and admin view state

### Backend
- Added `PersonView` struct for admin responses
- Updated `init_db` to create post_views table and enable foreign keys
- Added `get_admin_views` and `get_admin_view` handlers
- View counts ordered descending by default

### Frontend
- New admin panel page showing ID, Name, Email, Views columns
- Tab navigation with active state styling
- Auto-fetch views when switching to admin panel

### Database
- Created `/db` directory with schema management scripts
- Added `post_views` table with person_id FK and view_count
- Created shell scripts for schema management and SQLite CLI access

### Testing
- 4 unit tests covering table creation, view increment, cascade delete
- 10 integration tests covering full CRUD + view counter flows
- Playwright UI tests for page navigation and component visibility
- K6 stress tests with 4 scenarios and custom metrics

### Documentation
- Updated design-doc.md with implementation details
- Added feature documentation
- Created changes summary
- Updated changelog
- Enhanced README with new features

## [0.1.0] - Initial Release

### Added
- Person CRUD application with Rust/Actix-web backend
- SQLite database for person storage
- React frontend with create, read, update, delete operations
- REST API endpoints for person management
