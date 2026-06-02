package com.taxservice.domain;

public record TaxBracket(double rate, long lowerBound, long upperBound) {
}
