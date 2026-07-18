#!/usr/bin/env bash
set -euo pipefail
container="${1:-admin-console-demo-kafka}"
kafka() { podman exec "$container" "/opt/kafka/bin/$@"; }
existing=$(kafka kafka-topics.sh --bootstrap-server localhost:9092 --list 2>/dev/null || true)
for topic in orders.events payments.events audit.log; do
  case "$existing" in
    *"$topic"*) kafka kafka-topics.sh --bootstrap-server localhost:9092 --delete --topic "$topic" > /dev/null 2>&1 ;;
  esac
done
kafka kafka-topics.sh --bootstrap-server localhost:9092 --create --topic orders.events --partitions 3 --replication-factor 1 > /dev/null
kafka kafka-topics.sh --bootstrap-server localhost:9092 --create --topic payments.events --partitions 2 --replication-factor 1 > /dev/null
kafka kafka-topics.sh --bootstrap-server localhost:9092 --create --topic audit.log --partitions 1 --replication-factor 1 > /dev/null
for index in $(seq 1 240); do
  echo "order-$index:{\"orderId\":$index,\"status\":\"placed\",\"totalCents\":$(( 1000 + index * 37 ))}"
done | podman exec -i "$container" /opt/kafka/bin/kafka-console-producer.sh --bootstrap-server localhost:9092 --topic orders.events --property parse.key=true --property key.separator=:
for index in $(seq 1 120); do
  echo "payment-$index:{\"paymentId\":$index,\"method\":\"card\"}"
done | podman exec -i "$container" /opt/kafka/bin/kafka-console-producer.sh --bootstrap-server localhost:9092 --topic payments.events --property parse.key=true --property key.separator=:
for index in $(seq 1 60); do
  echo "audit-$index:{\"actor\":\"user$(( index % 7 ))\",\"action\":\"login\",\"seq\":$index}"
done | podman exec -i "$container" /opt/kafka/bin/kafka-console-producer.sh --bootstrap-server localhost:9092 --topic audit.log --property parse.key=true --property key.separator=:
podman exec "$container" timeout 15 /opt/kafka/bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic orders.events --group shop-workers --from-beginning --max-messages 60 > /dev/null 2>&1 || true
kafka kafka-get-offsets.sh --bootstrap-server localhost:9092 --topic orders.events
