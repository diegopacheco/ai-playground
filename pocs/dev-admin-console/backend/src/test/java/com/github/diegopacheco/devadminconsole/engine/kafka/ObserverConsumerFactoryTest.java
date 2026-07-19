package com.github.diegopacheco.devadminconsole.engine.kafka;

import static org.assertj.core.api.Assertions.assertThat;

import com.github.diegopacheco.devadminconsole.project.ConnectionConfig;
import com.github.diegopacheco.devadminconsole.project.ConnectionKind;
import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.Properties;
import org.apache.kafka.clients.consumer.ConsumerConfig;
import org.junit.jupiter.api.Test;

class ObserverConsumerFactoryTest {
    private final ObserverConsumerFactory factory = new ObserverConsumerFactory();
    private final ConnectionConfig config = new ConnectionConfig(1L, 1L, "kafka", ConnectionKind.KAFKA,
            "localhost", 9092, null, null, null, null, null, null, "tester");

    @Test
    void neverJoinsAConsumerGroupBecauseJoiningRebalancesEveryRealConsumerOfThatGroup() {
        Properties properties = factory.properties(config, 100);
        assertThat(properties.get(ConsumerConfig.GROUP_ID_CONFIG)).isNull();
    }

    @Test
    void disablesAutoCommitBecauseCommittingWouldCorruptRealConsumerProgressInProduction() {
        assertThat(factory.properties(config, 100).get(ConsumerConfig.ENABLE_AUTO_COMMIT_CONFIG)).isEqualTo(false);
    }

    @Test
    void usesAUniqueClientIdPerQuerySoTheConsoleIsIdentifiableAndNeverCollidesWithARealClient() {
        String first = String.valueOf(factory.properties(config, 100).get(ConsumerConfig.CLIENT_ID_CONFIG));
        String second = String.valueOf(factory.properties(config, 100).get(ConsumerConfig.CLIENT_ID_CONFIG));
        assertThat(first).startsWith("dev-admin-console-observer-").isNotEqualTo(second);
    }

    @Test
    void boundsEachPollSoAConsumeCannotPinABrokerConnectionIndefinitely() {
        Properties properties = factory.properties(config, 25);
        assertThat(properties.get(ConsumerConfig.MAX_POLL_RECORDS_CONFIG)).isEqualTo(25);
        assertThat(properties.get(ConsumerConfig.REQUEST_TIMEOUT_MS_CONFIG)).isEqualTo(10_000);
    }

    @Test
    void neverResetsOffsetsImplicitlySoTheConsoleReadsOnlyWhereItWasExplicitlyPointed() {
        assertThat(factory.properties(config, 100).get(ConsumerConfig.AUTO_OFFSET_RESET_CONFIG)).isEqualTo("none");
    }

    @Test
    void theEngineHasNoCommitCallPathAtAllWhichIsStrongerThanTrustingConfiguration() {
        boolean anyCommit = Arrays.stream(KafkaEngine.class.getDeclaredMethods())
                .map(Method::getName)
                .anyMatch(name -> name.toLowerCase().contains("commit"));
        assertThat(anyCommit).isFalse();
    }
}
