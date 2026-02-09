# Person CRUD Application

Full-stack person management application with view counter analytics.

## Architecture

- **Backend:** Rust + Actix-web + SQLite
- **Frontend:** React 18
- **Database:** SQLite with foreign key constraints

## Features

### Person CRUD
- Create, read, update, and delete person records
- Fields: name, email, age
- REST API with JSON responses

### View Counter
- Automatic view tracking on person detail requests
- Admin panel showing view counts per person
- View counts ordered by popularity
- Real-time refresh capability

## API Endpoints

### Person Management
- `GET /persons` - List all persons
- `GET /persons/{id}` - Get person by ID (increments view count)
- `POST /persons` - Create new person
- `PUT /persons/{id}` - Update person
- `DELETE /persons/{id}` - Delete person (cascades to view counts)

### Admin Analytics
- `GET /admin/views` - List all persons with view counts
- `GET /admin/views/{id}` - Get view count for specific person

## Database Schema

### persons table
```sql
CREATE TABLE persons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    age INTEGER NOT NULL
);
```

### post_views table
```sql
CREATE TABLE post_views (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER NOT NULL,
    view_count INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE
);
```

## Directory Structure

```
person-crud/
├── backend/          # Rust Actix-web server
│   ├── src/
│   │   ├── main.rs
│   │   └── lib.rs
│   ├── tests/        # Integration tests
│   └── Cargo.toml
├── frontend/         # React application
│   ├── src/
│   │   ├── App.js
│   │   └── index.js
│   ├── e2e/          # Playwright tests
│   └── package.json
├── db/               # Database schema and scripts
│   ├── schema.sql
│   ├── create-schema.sh
│   ├── run-sql-client.sh
│   ├── start-db.sh
│   └── stop-db.sh
├── k6/               # K6 stress tests
│   └── stress-test.js
├── design-doc.md     # Architecture documentation
└── todo.md           # Development workflow tracker
```

## Running the Application

### Backend
```bash
cd backend
cargo run
```
Server runs on http://localhost:8080

### Frontend
```bash
cd frontend
npm install
npm start
```
UI runs on http://localhost:3000

### Database
```bash
cd db
./create-schema.sh
./run-sql-client.sh
```

## Testing

### Unit Tests
```bash
cd backend
cargo test
```
4 tests covering database operations and view counter logic.

### Integration Tests
```bash
cd backend
cargo test --test integration_test
```
10 tests covering full API workflows.

### UI Tests (Playwright)
```bash
cd frontend
npx playwright install
npx playwright test
```
Tests navigation, form inputs, and admin panel.

### Stress Tests (K6)
```bash
k6 run k6/stress-test.js
```
Tests create, read, update, delete, and admin views under load.

## Development Workflow

See `todo.md` for the complete development process:
- Phase 1: Build (Backend, Frontend, Database)
- Phase 2: Test (Unit, Integration, UI, Stress)
- Phase 3: Review (Code, Security)
- Phase 4: Documentation (Design Doc, Features, Changes, Changelog, README)

## Dependencies

### Backend
- actix-web 4
- actix-cors 0.7
- serde 1
- rusqlite 0.31

### Frontend
- react 18
- react-dom 18
- playwright (testing)

## License

MIT
