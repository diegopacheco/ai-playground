# Deployment Guide

## Production Deployment Checklist

### Security

1. Generate a strong JWT secret:
```bash
openssl rand -base64 32
```

2. Update environment variables:
```bash
export JWT_SECRET="your-strong-random-secret"
export DATABASE_URL="postgresql://user:password@host:5432/dbname?sslmode=require"
export RUST_LOG="twitter_clone=info,tower_http=info"
```

3. Enable PostgreSQL SSL connections

4. Configure CORS for specific origins in `src/main.rs`:
```rust
let cors = CorsLayer::new()
    .allow_origin("https://yourdomain.com".parse::<HeaderValue>().unwrap())
    .allow_methods([Method::GET, Method::POST, Method::PUT, Method::DELETE])
    .allow_headers([AUTHORIZATION, CONTENT_TYPE]);
```

5. Use HTTPS in production

### Build Optimization

1. Build with release profile:
```bash
cargo build --release
```

2. Strip binary (optional):
```bash
strip target/release/twitter-clone
```

3. Binary location:
```
target/release/twitter-clone
```

### Database Setup

1. Create production database:
```sql
CREATE DATABASE twitter_prod;
```

2. Create database user:
```sql
CREATE USER twitter_app WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE twitter_prod TO twitter_app;
```

3. Run migrations:
```bash
DATABASE_URL="postgresql://twitter_app:password@host/twitter_prod" \
./target/release/twitter-clone
```

Migrations run automatically on server startup.

### Environment Configuration

Create `.env` file or set environment variables:

```bash
DATABASE_URL=postgresql://twitter_app:password@host:5432/twitter_prod?sslmode=require
JWT_SECRET=your-strong-random-secret-here
RUST_LOG=twitter_clone=info,tower_http=info
```

### Docker/Podman Deployment

Create `Containerfile`:

```dockerfile
FROM rust:1.93-alpine AS builder
RUN apk add --no-install-deps musl-dev postgresql-dev
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY migrations ./migrations
RUN cargo build --release

FROM alpine:latest
RUN apk add --no-install-deps libpq ca-certificates
WORKDIR /app
COPY --from=builder /app/target/release/twitter-clone .
COPY --from=builder /app/migrations ./migrations
EXPOSE 8000
ENV RUST_LOG=twitter_clone=info
CMD ["./twitter-clone"]
```

Build and run:
```bash
podman build -t twitter-clone:latest -f Containerfile .
podman run -d \
  -p 8000:8000 \
  -e DATABASE_URL="postgresql://..." \
  -e JWT_SECRET="..." \
  --name twitter-backend \
  twitter-clone:latest
```

### Kubernetes Deployment

Create `deployment.yaml`:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: twitter-secrets
type: Opaque
stringData:
  DATABASE_URL: "postgresql://user:pass@postgres-service:5432/twitter"
  JWT_SECRET: "your-secret-key"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: twitter-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: twitter-backend
  template:
    metadata:
      labels:
        app: twitter-backend
    spec:
      containers:
      - name: twitter-backend
        image: twitter-clone:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: twitter-secrets
              key: DATABASE_URL
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: twitter-secrets
              key: JWT_SECRET
        - name: RUST_LOG
          value: "twitter_clone=info,tower_http=info"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /api/auth/login
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: twitter-backend-service
spec:
  selector:
    app: twitter-backend
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

Deploy:
```bash
kubectl apply -f deployment.yaml
```

### Systemd Service

Create `/etc/systemd/system/twitter-backend.service`:

```ini
[Unit]
Description=Twitter Clone Backend
After=network.target postgresql.service

[Service]
Type=simple
User=twitter
WorkingDirectory=/opt/twitter-backend
Environment="DATABASE_URL=postgresql://..."
Environment="JWT_SECRET=..."
Environment="RUST_LOG=twitter_clone=info"
ExecStart=/opt/twitter-backend/twitter-clone
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable twitter-backend
sudo systemctl start twitter-backend
sudo systemctl status twitter-backend
```

### Reverse Proxy (Nginx)

Create `/etc/nginx/sites-available/twitter`:

```nginx
upstream twitter_backend {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name api.yourdomain.com;

    location / {
        return 301 https://$server_name$request_uri;
    }
}

server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/api.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.yourdomain.com/privkey.pem;

    location / {
        proxy_pass http://twitter_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }
}
```

Enable:
```bash
sudo ln -s /etc/nginx/sites-available/twitter /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### Monitoring

1. Add health check endpoint in `src/main.rs`:
```rust
async fn health_check() -> StatusCode {
    StatusCode::OK
}

let app = Router::new()
    .route("/health", get(health_check))
    .merge(routes::create_routes(state))
```

2. Set up monitoring with Prometheus:
```bash
curl http://localhost:8000/health
```

3. Log aggregation:
```bash
journalctl -u twitter-backend -f
```

### Database Backup

1. Automated backup script:
```bash
#!/bin/bash
pg_dump twitter_prod | gzip > backup-$(date +%Y%m%d-%H%M%S).sql.gz
```

2. Cron job:
```
0 2 * * * /opt/scripts/backup-db.sh
```

### Performance Tuning

1. Increase connection pool in `src/main.rs`:
```rust
let pool = PgPoolOptions::new()
    .max_connections(20)
    .connect(&config.database_url)
    .await?;
```

2. PostgreSQL tuning in `postgresql.conf`:
```
max_connections = 100
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
```

3. Add database indexes for query optimization

### Scaling

1. Horizontal scaling: Run multiple instances behind load balancer

2. Database read replicas: Configure PostgreSQL replication

3. Caching: Add Redis for frequently accessed data

4. CDN: Use CDN for static assets

### Monitoring and Alerts

1. Set up error tracking (Sentry, etc.)

2. Monitor metrics:
   - Request rate
   - Response time
   - Error rate
   - Database connections
   - Memory usage
   - CPU usage

3. Set up alerts for:
   - High error rate
   - Slow response times
   - Database connection pool exhaustion
   - High memory/CPU usage

### Maintenance

1. Log rotation:
```bash
logrotate /etc/logrotate.d/twitter-backend
```

2. Database vacuum:
```sql
VACUUM ANALYZE;
```

3. Update dependencies regularly:
```bash
cargo update
cargo audit
```

### Rollback Plan

1. Keep previous binary:
```bash
cp target/release/twitter-clone /opt/backups/twitter-clone-$(date +%Y%m%d)
```

2. Database backup before migrations

3. Test rollback procedure

### Security Best Practices

1. Keep dependencies updated
2. Run security audits: `cargo audit`
3. Use firewalls to restrict access
4. Regular security patches
5. Monitor logs for suspicious activity
6. Implement rate limiting
7. Use strong passwords and secrets
8. Enable database SSL
9. Regular backups
10. Disaster recovery plan

### Testing Production

1. Health check:
```bash
curl https://api.yourdomain.com/health
```

2. API test:
```bash
curl https://api.yourdomain.com/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"test","email":"test@test.com","password":"password123"}'
```

3. Load testing:
```bash
ab -n 1000 -c 10 https://api.yourdomain.com/health
```

### Troubleshooting

1. Check logs:
```bash
journalctl -u twitter-backend -n 100
```

2. Check database connections:
```sql
SELECT * FROM pg_stat_activity;
```

3. Check system resources:
```bash
htop
```

4. Test database connection:
```bash
psql $DATABASE_URL
```

## Post-Deployment

1. Monitor logs for errors
2. Check performance metrics
3. Verify all endpoints work
4. Test authentication flow
5. Monitor database performance
6. Set up alerts
7. Document any issues
8. Plan for scaling

## Maintenance Schedule

- Daily: Check logs and metrics
- Weekly: Review performance, check disk space
- Monthly: Update dependencies, security audit
- Quarterly: Load testing, disaster recovery test
