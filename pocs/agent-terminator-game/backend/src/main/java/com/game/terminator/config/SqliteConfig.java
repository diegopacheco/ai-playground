package com.game.terminator.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.jdbc.core.dialect.JdbcDialect;
import org.springframework.data.jdbc.repository.config.AbstractJdbcConfiguration;

@Configuration
public class SqliteConfig extends AbstractJdbcConfiguration {
    @Bean
    public JdbcDialect jdbcDialect() {
        return SqliteDialect.INSTANCE;
    }
}
