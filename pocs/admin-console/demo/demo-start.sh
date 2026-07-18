#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
backend="${BACKEND:-http://localhost:8099}"
cookies=/tmp/admin-console-demo-cookies.txt

if ! podman info > /dev/null 2>&1; then
  podman machine start
  for attempt in $(seq 1 60); do
    podman info > /dev/null 2>&1 && break
    sleep 1
  done
fi

podman-compose up -d > /dev/null
echo "waiting for target servers"

wait_for() {
  name="$1"
  shift
  for attempt in $(seq 1 120); do
    if "$@" > /dev/null 2>&1; then
      echo "  $name ready"
      return 0
    fi
    sleep 1
  done
  echo "  $name FAILED"
  return 1
}

wait_for postgres podman exec admin-console-demo-postgres pg_isready -U postgres
wait_for mysql podman exec admin-console-demo-mysql mysqladmin ping -uroot -proot
wait_for redis podman exec admin-console-demo-redis redis-cli ping
wait_for etcd podman exec admin-console-demo-etcd etcdctl endpoint health
wait_for cassandra podman exec admin-console-demo-cassandra cqlsh -e "SELECT now() FROM system.local"
wait_for kafka podman exec admin-console-demo-kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --list
wait_for elasticsearch curl -fsS http://localhost:9200

if [ "${RESEED:-yes}" = "no" ]; then
  echo "skipping seed (RESEED=no)"
else
  echo "seeding"
podman exec -i admin-console-demo-postgres psql -U postgres -d shop < seed/postgres.sql > /dev/null
podman exec -i admin-console-demo-mysql mysql -uroot -proot shop < seed/mysql.sql 2> /dev/null
podman cp seed/cassandra.cql admin-console-demo-cassandra:/tmp/schema.cql
podman exec admin-console-demo-cassandra cqlsh -f /tmp/schema.cql > /dev/null
./seed/cassandra-rows.sh > /dev/null
./seed/redis.sh > /dev/null
./seed/etcd.sh > /dev/null
./seed/kafka.sh > /dev/null
./seed/elastic.sh > /dev/null
  echo "  seeded"
fi

if ! curl -fsS "$backend/actuator/health" > /dev/null 2>&1; then
  echo "backend is not running on $backend, start it with ../start.sh"
  exit 1
fi

rm -f "$cookies"
curl -fsS -c "$cookies" -X POST "$backend/api/auth/login" \
  -H 'Content-Type: application/json' \
  -d "{\"username\":\"${ADMIN_USER:-admin}\",\"password\":\"${ADMIN_PASSWORD:-admin}\"}" > /dev/null

existing=$(curl -fsS -b "$cookies" "$backend/api/projects")
case "$existing" in
  *'"name":"demo"'*)
    project=$(printf '%s' "$existing" | sed -n 's/.*{"id":\([0-9]*\),"name":"demo".*/\1/p')
    ;;
  *)
    project=$(curl -fsS -b "$cookies" -X POST "$backend/api/projects" \
      -H 'Content-Type: application/json' -d '{"name":"demo"}' | sed -n 's/.*"id":\([0-9]*\).*/\1/p')
    ;;
esac

register() {
  curl -fsS -b "$cookies" -X POST "$backend/api/projects/$project/connections" \
    -H 'Content-Type: application/json' -d "$1" > /dev/null 2>&1 || true
}

register '{"name":"demo-postgres","kind":"postgres","host":"localhost","port":5432,"database":"shop","keyspace":"public","username":"console_reader","password":"console_reader"}'
register '{"name":"demo-mysql","kind":"mysql","host":"localhost","port":3306,"database":"shop","username":"console_reader","password":"console_reader"}'
register '{"name":"demo-cassandra","kind":"cassandra","host":"localhost","port":9042,"keyspace":"shop","datacenter":"datacenter1"}'
register '{"name":"demo-redis","kind":"redis","host":"localhost","port":6379}'
register '{"name":"demo-etcd","kind":"etcd","host":"localhost","port":2379}'
register '{"name":"demo-kafka","kind":"kafka","host":"localhost","port":9092}'
register '{"name":"demo-elasticsearch","kind":"elasticsearch","host":"localhost","port":9200}'

cat <<CONFIG

demo environment ready

console      http://localhost:4321
backend      $backend
swagger      $backend/swagger
login        ${ADMIN_USER:-admin} / ${ADMIN_PASSWORD:-admin}

project      demo (id $project)

postgres     localhost:5432  db=shop     console_reader / console_reader   (SELECT only)
mysql        localhost:3306  db=shop     console_reader / console_reader   (SELECT only)
cassandra    localhost:9042  keyspace=shop  datacenter1   no auth
redis        localhost:6379  no auth
etcd         localhost:2379  no auth
kafka        localhost:9092  no auth     topics: orders.events, payments.events, audit.log
elasticsearch localhost:9200 no auth     index: products (1000 docs, max_result_window 200)

root credentials for seeding only:
  postgres   postgres / postgres
  mysql      root / root

CONFIG
