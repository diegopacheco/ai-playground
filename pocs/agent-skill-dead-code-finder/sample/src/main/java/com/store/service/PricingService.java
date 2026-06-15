package com.store.service;

import com.store.model.OrderLine;
import com.store.util.MoneyUtils;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class PricingService {

    public long total(List<OrderLine> lines) {
        long sum = 0;
        for (OrderLine line : lines) {
            sum += line.unitPriceCents() * line.quantity();
        }
        return MoneyUtils.applyRate(sum, 1.0);
    }

    public long legacyDiscount(long cents) {
        return MoneyUtils.applyRate(cents, 0.85);
    }
}
