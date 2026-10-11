package com.github.diegopacheco.qdrant.amazon;

import java.util.List;

record Product(String id, String name, String category, String categoryPath, Double price, String about, String image, String url) {}

record Hit(String id, double score, String name, String category, Double price, String about, String image, String url) {}

record Category(String name, long count) {}

record Stats(long products, String collection) {}

record SearchRequest(String query, Integer limit, String category, Double minPrice, Double maxPrice) {}

record SearchResponse(List<Hit> hits) {}

record AskRequest(String question, String category, Double minPrice, Double maxPrice) {}

record AskResponse(String answer, List<Hit> sources) {}

record IngestRequest(Integer limit) {}

record IngestResponse(int indexed, int skipped) {}

record ErrorResponse(String error) {}
