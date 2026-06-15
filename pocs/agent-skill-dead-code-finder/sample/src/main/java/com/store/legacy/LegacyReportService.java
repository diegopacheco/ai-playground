package com.store.legacy;

import java.util.List;

public class LegacyReportService {

    private final CsvExporter exporter = new CsvExporter();

    public String build(List<String> orderIds) {
        return exporter.export(format(orderIds));
    }

    public List<String> format(List<String> orderIds) {
        return orderIds.stream().map(id -> "order:" + id).toList();
    }
}
