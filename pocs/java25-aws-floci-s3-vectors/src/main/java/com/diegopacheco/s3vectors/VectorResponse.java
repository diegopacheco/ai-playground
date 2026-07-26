package com.diegopacheco.s3vectors;

import java.util.List;

public record VectorResponse(String key, List<Float> values) {
}
