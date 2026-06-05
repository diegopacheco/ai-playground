package com.github.controlpanel.common;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class EncodingTest {

    @Test
    void labelsRoundTripPreservingNameAndColor() {
        String encoded = Encoding.encodeLabels(List.of(new Label("bug", "ff0000"), new Label("good first issue", "00ff00")));
        List<Label> decoded = Encoding.decodeLabels(encoded);
        assertEquals(2, decoded.size());
        assertEquals("bug", decoded.get(0).name());
        assertEquals("ff0000", decoded.get(0).color());
        assertEquals("good first issue", decoded.get(1).name());
        assertEquals("00ff00", decoded.get(1).color());
    }

    @Test
    void delimitersInLabelNamesAreNeutralizedSoDecodingStaysStable() {
        String encoded = Encoding.encodeLabels(List.of(new Label("area\tfrontend\nstuff", "abc")));
        List<Label> decoded = Encoding.decodeLabels(encoded);
        assertEquals(1, decoded.size());
        assertEquals("area frontend stuff", decoded.get(0).name());
    }

    @Test
    void emptyOrNullEncodesToNullAndDecodesToEmpty() {
        assertNull(Encoding.encodeLabels(List.of()));
        assertNull(Encoding.encodeList(List.of()));
        assertTrue(Encoding.decodeLabels(null).isEmpty());
        assertTrue(Encoding.decodeList(null).isEmpty());
    }

    @Test
    void listRoundTripDropsBlanks() {
        String encoded = Encoding.encodeList(List.of("alice", " ", "bob"));
        assertEquals(List.of("alice", "bob"), Encoding.decodeList(encoded));
    }
}
