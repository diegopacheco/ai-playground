package com.github.diegopacheco.qdrant.amazon;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

class RagServiceTest {

    private static final Hit KITE = new Hit("1", 0.8, "Rainbow Delta Kite", "Toys & Games", 14.5, "Easy to fly | Ages 5+", "", "");
    private static final Hit PUZZLE = new Hit("2", 0.7, "Wooden Farm Puzzle", "Toys & Games", null, "24 chunky pieces", "", "");

    @Test
    void promptGroundsTheModelOnEveryRetrievedProduct() {
        String prompt = RagService.prompt("gift for a 5 year old", List.of(KITE, PUZZLE));
        assertTrue(prompt.contains("1. Rainbow Delta Kite"));
        assertTrue(prompt.contains("Price: $14.50"));
        assertTrue(prompt.contains("Category: Toys & Games"));
        assertTrue(prompt.contains("About: Easy to fly | Ages 5+"));
        assertTrue(prompt.contains("2. Wooden Farm Puzzle"));
        assertTrue(prompt.contains("About: 24 chunky pieces"));
    }

    @Test
    void promptEndsWithTheQuestionSoTheModelAnswersIt() {
        String prompt = RagService.prompt("gift for a 5 year old", List.of(KITE));
        assertTrue(prompt.endsWith("Question: gift for a 5 year old\nAnswer:"));
    }

    @Test
    void promptForbidsRecommendingOutsideTheSourcesAndGivesTheFallbackAnswer() {
        String prompt = RagService.prompt("a laptop", List.of(KITE));
        assertTrue(prompt.contains("Recommend only products from the numbered list"));
        assertTrue(prompt.contains(RagService.NOTHING_FOUND));
    }

    @Test
    void missingPriceIsStatedAsUnknownSoTheModelDoesNotInventOne() {
        String prompt = RagService.prompt("puzzle", List.of(PUZZLE));
        assertTrue(prompt.contains("Price: unknown"));
        assertFalse(prompt.contains("null"));
    }

    @Test
    void longAboutTextIsCutSoFiveSourcesFitTheModelContext() {
        Hit verbose = new Hit("3", 0.5, "Verbose", "Toys & Games", 1.0, "x".repeat(RagService.ABOUT_LIMIT) + "TAIL", "", "");
        assertFalse(RagService.prompt("anything", List.of(verbose)).contains("TAIL"));
    }
}
