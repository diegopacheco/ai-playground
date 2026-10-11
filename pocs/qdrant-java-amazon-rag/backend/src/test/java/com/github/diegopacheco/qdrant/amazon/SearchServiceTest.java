package com.github.diegopacheco.qdrant.amazon;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class SearchServiceTest {

    @Test
    void noFiltersMeansNoConditionsSoTheWholeCatalogIsSearched() {
        assertTrue(SearchService.filter(null, null, null).isEmpty());
        assertTrue(SearchService.filter("  ", null, null).isEmpty());
    }

    @Test
    void categoryOnlyMatchesTheKeywordIndexedPayloadField() {
        assertEquals(
                List.of(Map.of("key", "category", "match", Map.of("value", "Toys & Games"))),
                SearchService.filter("Toys & Games", null, null));
    }

    @Test
    void priceRangeOnlyIsInclusiveOnBothEnds() {
        assertEquals(
                List.of(Map.of("key", "price", "range", Map.of("gte", 10.0, "lte", 50.0))),
                SearchService.filter(null, 10.0, 50.0));
    }

    @Test
    void openEndedPriceRangeSendsOnlyTheBoundThatWasGiven() {
        assertEquals(List.of(Map.of("key", "price", "range", Map.of("gte", 10.0))), SearchService.filter(null, 10.0, null));
        assertEquals(List.of(Map.of("key", "price", "range", Map.of("lte", 50.0))), SearchService.filter(null, null, 50.0));
    }

    @Test
    void categoryAndPriceMustBothHold() {
        assertEquals(
                List.of(
                        Map.of("key", "category", "match", Map.of("value", "Toys & Games")),
                        Map.of("key", "price", "range", Map.of("gte", 5.0, "lte", 20.0))),
                SearchService.filter("Toys & Games", 5.0, 20.0));
    }
}
