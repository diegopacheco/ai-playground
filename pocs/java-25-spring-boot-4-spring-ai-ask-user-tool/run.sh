#!/bin/bash
mvn clean package -DskipTests

nohup java -jar target/java-25-spring-boot-4-spring-ai-ask-user-tool-1.0-SNAPSHOT.jar > app.log 2>&1 &
echo $! > app.pid

MAX=60
COUNT=0
until grep -q "Started AskUserToolApplication" app.log 2>/dev/null; do
    sleep 1
    COUNT=$((COUNT + 1))
    if [ $COUNT -ge $MAX ]; then
        echo "App failed to start"
        cat app.log
        exit 1
    fi
done
echo "App running on port 8080"
