package com.github.diegopacheco.adminconsole.auth;

import com.github.diegopacheco.adminconsole.crypto.MasterKeyProvider;
import com.github.diegopacheco.adminconsole.crypto.PostgresKeyStore;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.time.Instant;
import java.util.Base64;
import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

@Component
public class Jwt {
    public record Principal(String username, String role) {}

    private static final Base64.Encoder ENCODER = Base64.getUrlEncoder().withoutPadding();
    private static final Base64.Decoder DECODER = Base64.getUrlDecoder();
    private static final long LIFETIME_SECONDS = 28_800;

    private final MasterKeyProvider keys;
    private final long lifetimeSeconds;

    @Autowired
    public Jwt(MasterKeyProvider keys) {
        this(keys, LIFETIME_SECONDS);
    }

    Jwt(MasterKeyProvider keys, long lifetimeSeconds) {
        this.keys = keys;
        this.lifetimeSeconds = lifetimeSeconds;
    }

    public String issue(String username, String role) {
        long now = Instant.now().getEpochSecond();
        String header = encode("{\"alg\":\"HS256\",\"typ\":\"JWT\"}");
        String claims = encode("{\"sub\":\"" + username + "\",\"role\":\"" + role + "\",\"iat\":" + now + ",\"exp\":" + (now + lifetimeSeconds) + "}");
        String content = header + "." + claims;
        return content + "." + ENCODER.encodeToString(sign(content));
    }

    public Principal verify(String token) {
        try {
            String[] parts = token.split("\\.", -1);
            if (parts.length != 3) {
                return null;
            }
            if (!MessageDigest.isEqual(sign(parts[0] + "." + parts[1]), DECODER.decode(parts[2]))) {
                return null;
            }
            String claims = new String(DECODER.decode(parts[1]), StandardCharsets.UTF_8);
            if (expiration(claims) <= Instant.now().getEpochSecond()) {
                return null;
            }
            return new Principal(claim(claims, "sub"), claim(claims, "role"));
        } catch (Exception error) {
            return null;
        }
    }

    public long lifetimeSeconds() {
        return lifetimeSeconds;
    }

    private long expiration(String claims) {
        int start = claims.indexOf("\"exp\":") + 6;
        int end = start;
        while (end < claims.length() && Character.isDigit(claims.charAt(end))) {
            end++;
        }
        return Long.parseLong(claims.substring(start, end));
    }

    private String claim(String claims, String name) {
        String marker = "\"" + name + "\":\"";
        int start = claims.indexOf(marker);
        if (start < 0) {
            return null;
        }
        start += marker.length();
        return claims.substring(start, claims.indexOf('"', start));
    }

    private byte[] sign(String content) {
        try {
            Mac mac = Mac.getInstance("HmacSHA256");
            mac.init(new SecretKeySpec(keys.key(PostgresKeyStore.JWT), "HmacSHA256"));
            return mac.doFinal(content.getBytes(StandardCharsets.UTF_8));
        } catch (Exception error) {
            throw new IllegalStateException("token could not be signed", error);
        }
    }

    private String encode(String value) {
        return ENCODER.encodeToString(value.getBytes(StandardCharsets.UTF_8));
    }
}
