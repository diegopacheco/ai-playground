package com.diegopacheco.s3vectors;

import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.NotNull;
import java.util.List;

public record VectorRequest(@NotEmpty List<@NotNull Float> values) {
}
