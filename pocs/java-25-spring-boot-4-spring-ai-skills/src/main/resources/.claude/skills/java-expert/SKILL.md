---
name: java-expert
description: Expert knowledge of Java 25 features including virtual threads, pattern matching,
  records, sealed classes, value types, and Project Loom/Valhalla features.
  Use when the user asks about Java programming, Java 25 features, concurrency,
  performance optimization, or modern Java idioms.
allowed-tools: Read, Grep
---

# Java 25 Expert Skill

## Overview
Provides expert guidance on Java 25 language features and best practices.

## Key Java 25 Features

### Virtual Threads (Project Loom)
- Use `Thread.ofVirtual().start(runnable)` for lightweight concurrency
- Virtual threads are ideal for I/O-bound workloads
- Use `Executors.newVirtualThreadPerTaskExecutor()` for thread pools

### Pattern Matching
- Switch expressions with pattern matching
- Deconstruction patterns for records
- Guarded patterns with `when` clauses

### Records
- Immutable data carriers with auto-generated constructors, equals, hashCode, toString
- Custom compact constructors for validation

### Sealed Classes
- Restrict class hierarchies with `sealed`, `permits`
- Enable exhaustive pattern matching

### String Templates (Preview)
- Template processors like `STR."\{value}"`
- Custom template processors

## Best Practices
- Prefer records over plain POJOs for data transfer
- Use sealed interfaces for algebraic data types
- Leverage virtual threads for high-throughput servers
- Use pattern matching to eliminate instanceof casts

## Examples

```java
record Point(int x, int y) {}

sealed interface Shape permits Circle, Rectangle {}
record Circle(Point center, double radius) implements Shape {}
record Rectangle(Point topLeft, Point bottomRight) implements Shape {}

String describe(Shape shape) {
    return switch (shape) {
        case Circle c when c.radius() > 10 -> "Large circle";
        case Circle c -> "Small circle";
        case Rectangle r -> "Rectangle";
    };
}
```
