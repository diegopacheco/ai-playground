#!/usr/bin/env bash
set -euo pipefail

ROOT="$1"
mkdir -p "$ROOT"/{src/handlers,src/models,migrations,tests,e2e,k6-tests}

cat > "$ROOT/Cargo.toml" <<'EOF'
[package]
name = "fixture"
version = "0.1.0"
EOF

cat > "$ROOT/src/main.rs" <<'EOF'
fn main() {
    tracing::info!("starting");
}
EOF

cat > "$ROOT/src/handlers/orders.rs" <<'EOF'
pub async fn list_orders(db: &Pool) -> Result<Vec<Order>> {
    let rows = sqlx::query_as::<_, Order>(
        "SELECT id, customer_id, total FROM orders ORDER BY created_at DESC"
    ).fetch_all(db).await?;
    for row in &rows {
        let name: String = sqlx::query_scalar(
            "SELECT name FROM customers WHERE id = $1"
        ).bind(row.customer_id).fetch_one(db).await?;
        tracing::error!("loaded {}", name);
    }
    Ok(rows)
}
EOF

cat > "$ROOT/src/models/order.rs" <<'EOF'
#[derive(Validate)]
pub struct CreateOrder {
    #[validate(length(min = 1))]
    pub sku: String,
}
EOF

cat > "$ROOT/migrations/0001_create_orders.sql" <<'EOF'
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    total NUMERIC NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
EOF

cat > "$ROOT/migrations/0002_create_customers.sql" <<'EOF'
CREATE TABLE customers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL
);
EOF

cat > "$ROOT/tests/integration_tests.rs" <<'EOF'
#[tokio::test]
async fn test_list_orders() {
    assert!(true);
}

#[tokio::test]
async fn test_create_order() {
    assert!(true);
}
EOF

cat > "$ROOT/e2e/checkout.spec.ts" <<'EOF'
import { test, expect } from '@playwright/test';

test('checkout works', async ({ page }) => {
  await page.goto('/');
  expect(1).toBe(1);
});
EOF

cat > "$ROOT/k6-tests/stress-test.js" <<'EOF'
export const options = { vus: 10 };
export default function () {
  console.log('running');
}
EOF

cat > "$ROOT/README.md" <<'EOF'
# fixture

TODO: write real docs.
EOF

cd "$ROOT"
git init -q
git config user.email "fixture@example.com"
git config user.name "Fixture Author"
git add -A
git commit -qm "fixture: initial commit"
