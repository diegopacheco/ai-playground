# Design Doc: Post View Counter

## Architecture Overview

The existing Person CRUD application (Rust/Actix-web backend + React frontend + SQLite) will be extended with a post/person view tracking system. Every time a person record is viewed (GET /persons/{id}), the view count increments. An admin panel page displays all persons with their total view counts.

## Backend API Endpoints

### Existing Endpoints (unchanged)
- `GET /persons` - List all persons
- `GET /persons/{id}` - Get person by ID (now also increments view count)
- `POST /persons` - Create person
- `PUT /persons/{id}` - Update person
- `DELETE /persons/{id}` - Delete person

### New Endpoints
- `GET /admin/views` - Returns all persons with their view counts (id, name, email, view_count)
- `GET /admin/views/{id}` - Returns view count for a specific person

### Backend Responsibilities
- Increment view count atomically on each `GET /persons/{id}` call
- Serve aggregated view data for the admin panel
- Initialize the post_views table on startup alongside persons table

## Frontend Components

### Existing
- `App.js` - Main CRUD component (unchanged)

### New
- `AdminPanel.js` - Admin panel page showing view counts per person in a table
- Navigation between CRUD view and Admin Panel via tabs/buttons at the top

### Interactions
- Admin panel fetches from `GET /admin/views` on mount
- Displays a table with columns: ID, Name, Email, Views
- Auto-refreshes or has a refresh button

## Database Schema Design

### Existing Table
```sql
persons (id INTEGER PK, name TEXT, email TEXT, age INTEGER)
```

### New Table
```sql
post_views (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER NOT NULL,
    view_count INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE
)
```

### Alternative (simpler): Single row per person in post_views
- On first view: INSERT with view_count=1
- On subsequent views: UPDATE view_count = view_count + 1
- Uses UPSERT (INSERT OR REPLACE / ON CONFLICT)

## Integration Points

- Backend `get_person` handler increments view count in post_views table before returning person data
- Backend `create_person` handler initializes a post_views row with view_count=0
- Backend `delete_person` handler cascades to delete post_views row
- Frontend Admin Panel calls `/admin/views` REST endpoint
- Frontend navigation switches between CRUD view and Admin Panel
