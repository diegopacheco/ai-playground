package com.github.diegopacheco.devadminconsole.crypto;

import java.nio.charset.StandardCharsets;
import java.security.SecureRandom;
import java.util.Arrays;
import javax.crypto.Cipher;
import javax.crypto.spec.GCMParameterSpec;
import javax.crypto.spec.SecretKeySpec;
import org.springframework.stereotype.Component;

@Component
public class SecretCipher {
    private static final int IV_BYTES = 12;
    private static final int TAG_BITS = 128;

    private final MasterKeyProvider keys;
    private final SecureRandom random = new SecureRandom();

    public SecretCipher(MasterKeyProvider keys) {
        this.keys = keys;
    }

    public byte[] encrypt(String plaintext) {
        if (plaintext == null) {
            return null;
        }
        try {
            byte[] iv = new byte[IV_BYTES];
            random.nextBytes(iv);
            Cipher cipher = Cipher.getInstance("AES/GCM/NoPadding");
            cipher.init(Cipher.ENCRYPT_MODE, secretKey(), new GCMParameterSpec(TAG_BITS, iv));
            byte[] encrypted = cipher.doFinal(plaintext.getBytes(StandardCharsets.UTF_8));
            byte[] result = new byte[iv.length + encrypted.length];
            System.arraycopy(iv, 0, result, 0, iv.length);
            System.arraycopy(encrypted, 0, result, iv.length, encrypted.length);
            return result;
        } catch (Exception error) {
            throw new IllegalStateException("secret could not be encrypted", error);
        }
    }

    public String decrypt(byte[] ciphertext) {
        if (ciphertext == null) {
            return null;
        }
        if (ciphertext.length <= IV_BYTES) {
            throw new IllegalStateException("secret is malformed");
        }
        try {
            byte[] iv = Arrays.copyOfRange(ciphertext, 0, IV_BYTES);
            byte[] payload = Arrays.copyOfRange(ciphertext, IV_BYTES, ciphertext.length);
            Cipher cipher = Cipher.getInstance("AES/GCM/NoPadding");
            cipher.init(Cipher.DECRYPT_MODE, secretKey(), new GCMParameterSpec(TAG_BITS, iv));
            return new String(cipher.doFinal(payload), StandardCharsets.UTF_8);
        } catch (Exception error) {
            throw new IllegalStateException("secret could not be decrypted", error);
        }
    }

    private SecretKeySpec secretKey() {
        return new SecretKeySpec(keys.key(PostgresKeyStore.MASTER), "AES");
    }
}
