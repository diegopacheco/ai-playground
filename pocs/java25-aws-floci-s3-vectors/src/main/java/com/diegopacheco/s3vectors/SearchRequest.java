package com.diegopacheco.s3vectors;

import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.NotNull;
import java.util.List;

public record SearchRequest(
        @NotEmpty List<@NotNull Float> values,
        @NotNull @Min(1) @Max(100) Integer topK) {
}
