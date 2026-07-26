package com.diegopacheco.s3vectors;

import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.List;
import org.junit.jupiter.api.Test;
import org.springframework.web.server.ResponseStatusException;

class VectorServiceTest {

    private final VectorService service = new VectorService(
            null,
            "product-vectors",
            "products",
            3);

    @Test
    void rejectsWrongDimension() {
        assertThatThrownBy(() -> service.put("invalid", List.of(1.0f, 2.0f)))
                .isInstanceOf(ResponseStatusException.class);
    }

    @Test
    void rejectsZeroCosineVector() {
        assertThatThrownBy(() -> service.put("invalid", List.of(0.0f, 0.0f, 0.0f)))
                .isInstanceOf(ResponseStatusException.class);
    }
}
