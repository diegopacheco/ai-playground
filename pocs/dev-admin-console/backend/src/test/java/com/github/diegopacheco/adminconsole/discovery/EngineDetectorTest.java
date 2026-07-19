package com.github.diegopacheco.adminconsole.discovery;

import static org.assertj.core.api.Assertions.assertThat;

import com.github.diegopacheco.adminconsole.project.ConnectionKind;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

class EngineDetectorTest {
    private final EngineDetector detector = new EngineDetector();

    @ParameterizedTest
    @CsvSource({
            "docker.io/library/postgres:18, POSTGRES",
            "postgres:16-alpine, POSTGRES",
            "docker.io/library/mysql:9, MYSQL",
            "mariadb:11, MYSQL",
            "docker.io/library/cassandra:5.0, CASSANDRA",
            "docker.io/library/redis:8, REDIS",
            "valkey/valkey:8, REDIS",
            "quay.io/coreos/etcd:v3.6.6, ETCD",
            "apache/kafka:4.1.0, KAFKA",
            "docker.redpanda.com/redpandadata/redpanda:latest, KAFKA",
            "docker.elastic.co/elasticsearch/elasticsearch:9.1.2, ELASTICSEARCH",
            "opensearchproject/opensearch:2, ELASTICSEARCH"
    })
    void recognisesTheEnginesWeSupportFromTheirImageName(String image, ConnectionKind expected) {
        assertThat(detector.detect(image)).contains(expected);
    }

    @ParameterizedTest
    @CsvSource({"nginx:latest", "docker.io/library/busybox", "grafana/grafana:11", "hashicorp/vault:1.17"})
    void ignoresContainersThatAreNotAnEngineWeSupport(String image) {
        assertThat(detector.detect(image)).isEmpty();
    }

    @Test
    void ignoresMissingImageNamesRatherThanThrowing() {
        assertThat(detector.detect(null)).isEmpty();
        assertThat(detector.detect("")).isEmpty();
    }

    @Test
    void readsThePostgresDatabaseAndUserFromTheContainerEnvironment() {
        Map<String, String> env = Map.of("POSTGRES_DB", "shop", "POSTGRES_USER", "app", "POSTGRES_PASSWORD", "s3cr3t");
        assertThat(detector.database(ConnectionKind.POSTGRES, env)).isEqualTo("shop");
        assertThat(detector.username(ConnectionKind.POSTGRES, env)).isEqualTo("app");
        assertThat(detector.password(ConnectionKind.POSTGRES, env)).isEqualTo("s3cr3t");
    }

    @Test
    void fallsBackToThePostgresSuperuserWhenNoUserWasConfigured() {
        Map<String, String> env = Map.of("POSTGRES_PASSWORD", "postgres");
        assertThat(detector.username(ConnectionKind.POSTGRES, env)).isEqualTo("postgres");
        assertThat(detector.isSuperuser(ConnectionKind.POSTGRES, env)).isTrue();
    }

    @Test
    void prefersTheApplicationMysqlUserOverRootWhenTheContainerDefinesOne() {
        Map<String, String> env = Map.of("MYSQL_ROOT_PASSWORD", "root", "MYSQL_USER", "app",
                "MYSQL_PASSWORD", "app-pass", "MYSQL_DATABASE", "shop");
        assertThat(detector.username(ConnectionKind.MYSQL, env)).isEqualTo("app");
        assertThat(detector.password(ConnectionKind.MYSQL, env)).isEqualTo("app-pass");
        assertThat(detector.isSuperuser(ConnectionKind.MYSQL, env)).isFalse();
    }

    @Test
    void fallsBackToMysqlRootAndFlagsItAsSuperuserSoTheUiCanWarn() {
        Map<String, String> env = Map.of("MYSQL_ROOT_PASSWORD", "root", "MYSQL_DATABASE", "shop");
        assertThat(detector.username(ConnectionKind.MYSQL, env)).isEqualTo("root");
        assertThat(detector.password(ConnectionKind.MYSQL, env)).isEqualTo("root");
        assertThat(detector.isSuperuser(ConnectionKind.MYSQL, env)).isTrue();
    }

    @Test
    void leavesCredentialsEmptyForEnginesThatUsuallyRunWithoutAuth() {
        Map<String, String> env = Map.of();
        assertThat(detector.username(ConnectionKind.REDIS, env)).isNull();
        assertThat(detector.username(ConnectionKind.ETCD, env)).isNull();
        assertThat(detector.username(ConnectionKind.KAFKA, env)).isNull();
        assertThat(detector.isSuperuser(ConnectionKind.REDIS, env)).isFalse();
    }

    @Test
    void picksUpARedisPasswordWhenTheContainerSetsOne() {
        assertThat(detector.password(ConnectionKind.REDIS, Map.of("REDIS_PASSWORD", "hunter2"))).isEqualTo("hunter2");
    }

    @Test
    void defaultsPostgresToThePublicSchemaSoTheTreeIsNotEmptyOnImport() {
        assertThat(detector.keyspace(ConnectionKind.POSTGRES, Map.of())).isEqualTo("public");
        assertThat(detector.keyspace(ConnectionKind.REDIS, Map.of())).isNull();
    }

    @Test
    void knowsTheDefaultPortOfEveryEngineSoPortMatchingWorks() {
        assertThat(detector.defaultPort(ConnectionKind.POSTGRES)).isEqualTo(5432);
        assertThat(detector.defaultPort(ConnectionKind.MYSQL)).isEqualTo(3306);
        assertThat(detector.defaultPort(ConnectionKind.CASSANDRA)).isEqualTo(9042);
        assertThat(detector.defaultPort(ConnectionKind.REDIS)).isEqualTo(6379);
        assertThat(detector.defaultPort(ConnectionKind.ETCD)).isEqualTo(2379);
        assertThat(detector.defaultPort(ConnectionKind.KAFKA)).isEqualTo(9092);
        assertThat(detector.defaultPort(ConnectionKind.ELASTICSEARCH)).isEqualTo(9200);
    }
}
