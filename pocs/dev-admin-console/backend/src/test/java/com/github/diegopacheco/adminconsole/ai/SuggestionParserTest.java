package com.github.diegopacheco.adminconsole.ai;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Test;

class SuggestionParserTest {
    private final SuggestionParser parser = new SuggestionParser();

    @Test
    void takesTheQueryOutOfACodeFenceWhichIsHowModelsUsuallyAnswer() {
        String output = """
                Here is the query you asked for:

                ```sql
                SELECT count(*) FROM customers
                ```

                It counts every customer.
                """;
        assertThat(parser.extract(output)).isEqualTo("SELECT count(*) FROM customers");
    }

    @Test
    void keepsMultiLineQueriesIntact() {
        String output = """
                ```sql
                SELECT c.email, count(o.id)
                FROM customers c
                JOIN orders o ON o.customer_id = c.id
                GROUP BY c.email
                ```
                """;
        assertThat(parser.extract(output)).isEqualTo("""
                SELECT c.email, count(o.id)
                FROM customers c
                JOIN orders o ON o.customer_id = c.id
                GROUP BY c.email""");
    }

    @Test
    void takesTheFirstBlockWhenAModelOffersSeveralAlternatives() {
        String output = """
                ```sql
                SELECT 1
                ```
                or maybe
                ```sql
                SELECT 2
                ```
                """;
        assertThat(parser.extract(output)).isEqualTo("SELECT 1");
    }

    @Test
    void handlesAFenceWithNoLanguageTag() {
        assertThat(parser.extract("```\nget /config --prefix\n```")).isEqualTo("get /config --prefix");
    }

    @Test
    void fallsBackToTheWholeAnswerWhenTheModelSkipsTheFence() {
        assertThat(parser.extract("SELECT count(*) FROM customers")).isEqualTo("SELECT count(*) FROM customers");
    }

    @Test
    void trimsSurroundingBlankLinesSoTheEditorDoesNotOpenWithEmptyLines() {
        assertThat(parser.extract("\n\n  SELECT 1  \n\n")).isEqualTo("SELECT 1");
    }

    @Test
    void handlesAnUnclosedFenceRatherThanReturningNothing() {
        assertThat(parser.extract("```sql\nSELECT 1\n")).isEqualTo("SELECT 1");
    }

    @Test
    void failsLoudlyWhenTheAgentReturnsNothingSoTheUiDoesNotShowAnEmptySuggestion() {
        assertThatThrownBy(() -> parser.extract("")).isInstanceOf(IllegalStateException.class);
        assertThatThrownBy(() -> parser.extract(null)).isInstanceOf(IllegalStateException.class);
        assertThatThrownBy(() -> parser.extract("   \n  ")).isInstanceOf(IllegalStateException.class);
    }

    @Test
    void doesNotSanitiseTheStatementBecauseTheReadOnlyGuardIsWhatDecidesSafety() {
        assertThat(parser.extract("```sql\nDELETE FROM customers\n```")).isEqualTo("DELETE FROM customers");
    }
}
