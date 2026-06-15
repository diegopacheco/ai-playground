package com.store.legacy;

import java.util.List;

public class CsvExporter {

    public String export(List<String> rows) {
        StringBuilder out = new StringBuilder();
        for (String row : rows) {
            out.append(escape(row)).append('\n');
        }
        return out.toString();
    }

    public String escape(String value) {
        return value.replace(",", " ");
    }
}
