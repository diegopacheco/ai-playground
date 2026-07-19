package com.github.diegopacheco.devadminconsole.engine.kafka;

import com.github.diegopacheco.devadminconsole.project.ConnectionConfig;
import java.util.Properties;
import java.util.UUID;
import org.apache.kafka.clients.consumer.ConsumerConfig;
import org.apache.kafka.clients.consumer.KafkaConsumer;
import org.apache.kafka.common.serialization.StringDeserializer;
import org.springframework.stereotype.Component;

@Component
public class ObserverConsumerFactory {
    public Properties properties(ConnectionConfig config, int maxRecords) {
        Properties properties = new Properties();
        properties.put(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, config.host() + ":" + config.port());
        properties.put(ConsumerConfig.KEY_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class.getName());
        properties.put(ConsumerConfig.VALUE_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class.getName());
        properties.put(ConsumerConfig.ENABLE_AUTO_COMMIT_CONFIG, false);
        properties.put(ConsumerConfig.AUTO_OFFSET_RESET_CONFIG, "none");
        properties.put(ConsumerConfig.MAX_POLL_RECORDS_CONFIG, maxRecords);
        properties.put(ConsumerConfig.CLIENT_ID_CONFIG, "dev-admin-console-observer-" + UUID.randomUUID());
        properties.put(ConsumerConfig.REQUEST_TIMEOUT_MS_CONFIG, 10_000);
        properties.put(ConsumerConfig.DEFAULT_API_TIMEOUT_MS_CONFIG, 10_000);
        return properties;
    }

    public KafkaConsumer<String, String> create(ConnectionConfig config, int maxRecords) {
        return new KafkaConsumer<>(properties(config, maxRecords));
    }
}
