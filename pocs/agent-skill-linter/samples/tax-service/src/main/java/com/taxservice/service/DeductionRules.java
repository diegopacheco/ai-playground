package com.taxservice.service;

import com.taxservice.domain.FilingStatus;
import org.springframework.stereotype.Component;

@Component
public class DeductionRules {

    public long standardDeduction(FilingStatus status) {
        return switch (status) {
            case SINGLE -> 14600;
            case MARRIED_FILING_JOINTLY -> 29200;
            case HEAD_OF_HOUSEHOLD -> 21900;
        };
    }

    public long applicableDeduction(FilingStatus status, long itemizedDeductions) {
        long standard = standardDeduction(status);
        if (itemizedDeductions > standard) {
            return itemizedDeductions;
        }
        return standard;
    }
}
