package com.game.terminator.config;

import org.springframework.data.jdbc.core.dialect.JdbcDialect;
import org.springframework.data.relational.core.dialect.LimitClause;
import org.springframework.data.relational.core.dialect.LockClause;
import org.springframework.data.relational.core.sql.LockOptions;
import org.springframework.data.relational.core.sql.render.SelectRenderContext;

public class SqliteDialect implements JdbcDialect {

    public static final SqliteDialect INSTANCE = new SqliteDialect();

    @Override
    public LimitClause limit() {
        return new LimitClause() {
            @Override
            public String getLimit(long limit) {
                return "LIMIT " + limit;
            }

            @Override
            public String getOffset(long offset) {
                return "OFFSET " + offset;
            }

            @Override
            public String getLimitOffset(long limit, long offset) {
                return "LIMIT " + limit + " OFFSET " + offset;
            }

            @Override
            public Position getClausePosition() {
                return Position.AFTER_ORDER_BY;
            }
        };
    }

    @Override
    public LockClause lock() {
        return new LockClause() {
            @Override
            public String getLock(LockOptions lockOptions) {
                return "";
            }

            @Override
            public Position getClausePosition() {
                return Position.AFTER_ORDER_BY;
            }
        };
    }

    @Override
    public SelectRenderContext getSelectContext() {
        return new SelectRenderContext() {};
    }
}
