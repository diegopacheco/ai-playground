package com.store.model;

import java.util.List;

public record Order(String id, List<OrderLine> lines, long totalCents) {
}
