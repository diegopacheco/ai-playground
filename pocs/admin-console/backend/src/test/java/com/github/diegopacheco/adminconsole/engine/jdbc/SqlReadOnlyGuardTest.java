package com.github.diegopacheco.adminconsole.engine.jdbc;

import static org.assertj.core.api.Assertions.assertThatCode;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.github.diegopacheco.adminconsole.engine.ReadOnlyViolation;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

class SqlReadOnlyGuardTest {
    private final SqlReadOnlyGuard guard = new SqlReadOnlyGuard();

    @ParameterizedTest
    @ValueSource(strings = {
            "SELECT * FROM orders",
            "select id, name from customers where id = 1",
            "  \n  SELECT 1  ",
            "SHOW TABLES",
            "DESCRIBE orders",
            "EXPLAIN SELECT * FROM orders",
            "WITH recent AS (SELECT * FROM orders LIMIT 10) SELECT * FROM recent",
            "SELECT * FROM orders -- a trailing comment",
            "SELECT * FROM orders;"
    })
    void allowsReadStatementsSoTheConsoleIsActuallyUsable(String statement) {
        assertThatCode(() -> guard.assertReadOnly(statement)).doesNotThrowAnyException();
    }

    @ParameterizedTest
    @ValueSource(strings = {
            "DELETE FROM orders",
            "delete from orders",
            "INSERT INTO orders VALUES (1)",
            "UPDATE orders SET total = 0",
            "DROP TABLE orders",
            "TRUNCATE orders",
            "ALTER TABLE orders ADD COLUMN x INT",
            "CREATE TABLE t (id INT)",
            "GRANT ALL ON orders TO public"
    })
    void rejectsPlainWriteStatementsBecauseTheConsoleMustNeverMutateATarget(String statement) {
        assertThatThrownBy(() -> guard.assertReadOnly(statement)).isInstanceOf(ReadOnlyViolation.class);
    }

    @Test
    void rejectsASecondStatementSmuggledAfterASemicolonWhichIsTheClassicConsoleBypass() {
        assertThatThrownBy(() -> guard.assertReadOnly("SELECT 1; DROP TABLE orders"))
                .isInstanceOf(ReadOnlyViolation.class)
                .hasMessageContaining("multiple statements");
    }

    @Test
    void rejectsAWriteHiddenInsideACteBecauseThePostgresCteCanActuallyDelete() {
        assertThatThrownBy(() -> guard.assertReadOnly(
                "WITH gone AS (DELETE FROM orders RETURNING *) SELECT * FROM gone"))
                .isInstanceOf(ReadOnlyViolation.class);
    }

    @Test
    void rejectsAWriteObfuscatedByAnInlineCommentSinceTheServerIgnoresTheComment() {
        assertThatThrownBy(() -> guard.assertReadOnly("SEL/**/ECT 1")).isInstanceOf(ReadOnlyViolation.class);
        assertThatThrownBy(() -> guard.assertReadOnly("DEL/**/ETE FROM orders")).isInstanceOf(ReadOnlyViolation.class);
    }

    @Test
    void rejectsAWriteHiddenBehindALeadingCommentBlock() {
        assertThatThrownBy(() -> guard.assertReadOnly("/* harmless */ DELETE FROM orders"))
                .isInstanceOf(ReadOnlyViolation.class);
        assertThatThrownBy(() -> guard.assertReadOnly("-- harmless\nDROP TABLE orders"))
                .isInstanceOf(ReadOnlyViolation.class);
    }

    @Test
    void rejectsExplainAnalyzeBecauseItExecutesTheStatementItIsExplaining() {
        assertThatThrownBy(() -> guard.assertReadOnly("EXPLAIN ANALYZE DELETE FROM orders"))
                .isInstanceOf(ReadOnlyViolation.class);
        assertThatThrownBy(() -> guard.assertReadOnly("EXPLAIN ANALYZE SELECT * FROM orders"))
                .isInstanceOf(ReadOnlyViolation.class)
                .hasMessageContaining("executes");
    }

    @Test
    void rejectsSelectIntoOutfileWhichWritesToTheServerFilesystem() {
        assertThatThrownBy(() -> guard.assertReadOnly("SELECT * FROM orders INTO OUTFILE '/tmp/x'"))
                .isInstanceOf(ReadOnlyViolation.class);
    }

    @Test
    void rejectsSetSoSessionStateIncludingReadOnlyModeCannotBeChanged() {
        assertThatThrownBy(() -> guard.assertReadOnly("SET default_transaction_read_only = off"))
                .isInstanceOf(ReadOnlyViolation.class);
    }

    @Test
    void rejectsAnEmptyStatementRatherThanPassingItToTheDriver() {
        assertThatThrownBy(() -> guard.assertReadOnly("   ")).isInstanceOf(ReadOnlyViolation.class);
        assertThatThrownBy(() -> guard.assertReadOnly(null)).isInstanceOf(ReadOnlyViolation.class);
    }

    @Test
    void allowsASemicolonInsideAStringLiteralSinceItIsDataNotAStatementBreak() {
        assertThatCode(() -> guard.assertReadOnly("SELECT * FROM orders WHERE note = 'a;b'"))
                .doesNotThrowAnyException();
    }

    @Test
    void allowsAWriteWordInsideAStringLiteralSinceItIsDataNotAnOperation() {
        assertThatCode(() -> guard.assertReadOnly("SELECT * FROM audit WHERE action = 'delete'"))
                .doesNotThrowAnyException();
    }
}
