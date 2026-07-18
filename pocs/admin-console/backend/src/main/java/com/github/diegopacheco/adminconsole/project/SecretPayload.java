package com.github.diegopacheco.adminconsole.project;

import com.github.diegopacheco.adminconsole.crypto.SecretCipher;
import java.util.Map;
import org.springframework.stereotype.Component;
import tools.jackson.databind.ObjectMapper;

@Component
public class SecretPayload {
    private final SecretCipher cipher;
    private final ObjectMapper mapper = new ObjectMapper();

    public SecretPayload(SecretCipher cipher) {
        this.cipher = cipher;
    }

    public byte[] seal(String password) {
        if (password == null || password.isEmpty()) {
            return null;
        }
        return cipher.encrypt(mapper.writeValueAsString(Map.of("password", password)));
    }

    public String openPassword(byte[] ciphertext) {
        if (ciphertext == null) {
            return null;
        }
        Map<String, Object> values = mapper.readValue(cipher.decrypt(ciphertext), Map.class);
        Object password = values.get("password");
        return password == null ? null : String.valueOf(password);
    }
}
