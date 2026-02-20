#!/bin/bash
mvn clean package -DskipTests
java -jar target/java-25-spring-boot-4-spring-ai-ask-user-tool-1.0-SNAPSHOT.jar
