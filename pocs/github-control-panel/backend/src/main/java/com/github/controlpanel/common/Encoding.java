package com.github.controlpanel.common;

import java.util.ArrayList;
import java.util.List;

public final class Encoding {

    public static final String FIELD = "\t";
    public static final String RECORD = "\n";

    private Encoding() {
    }

    public static String encodeLabels(List<Label> labels) {
        if (labels == null || labels.isEmpty()) {
            return null;
        }
        List<String> parts = new ArrayList<>();
        for (Label label : labels) {
            parts.add(clean(label.name()) + FIELD + clean(label.color()));
        }
        return String.join(RECORD, parts);
    }

    public static String encodeList(List<String> items) {
        if (items == null || items.isEmpty()) {
            return null;
        }
        List<String> parts = new ArrayList<>();
        for (String item : items) {
            if (item != null && !item.isBlank()) {
                parts.add(clean(item));
            }
        }
        return parts.isEmpty() ? null : String.join(RECORD, parts);
    }

    public static List<Label> decodeLabels(String value) {
        List<Label> labels = new ArrayList<>();
        if (value == null || value.isEmpty()) {
            return labels;
        }
        for (String record : value.split(RECORD)) {
            String[] parts = record.split(FIELD, 2);
            String name = parts[0];
            String color = parts.length > 1 ? parts[1] : "";
            labels.add(new Label(name, color));
        }
        return labels;
    }

    public static List<String> decodeList(String value) {
        List<String> items = new ArrayList<>();
        if (value == null || value.isEmpty()) {
            return items;
        }
        for (String item : value.split(RECORD)) {
            if (!item.isEmpty()) {
                items.add(item);
            }
        }
        return items;
    }

    private static String clean(String value) {
        if (value == null) {
            return "";
        }
        return value.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ');
    }
}
