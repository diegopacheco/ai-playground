package com.github.diegopacheco.adminconsole.crypto;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.Arrays;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class SecretCipherTest {
    private SecretCipher cipher;

    @BeforeEach
    void setUp() {
        byte[] key = new byte[32];
        Arrays.fill(key, (byte) 7);
        cipher = new SecretCipher(purpose -> key);
    }

    @Test
    void roundTripsASecretSoStoredPasswordsCanBeUsedToConnect() {
        assertThat(cipher.decrypt(cipher.encrypt("s3cr3t"))).isEqualTo("s3cr3t");
    }

    @Test
    void producesDifferentCiphertextForTheSameSecretSoRepeatedPasswordsAreNotCorrelatable() {
        byte[] first = cipher.encrypt("same");
        byte[] second = cipher.encrypt("same");
        assertThat(first).isNotEqualTo(second);
        assertThat(cipher.decrypt(first)).isEqualTo(cipher.decrypt(second));
    }

    @Test
    void rejectsTamperedCiphertextRatherThanReturningGarbage() {
        byte[] ciphertext = cipher.encrypt("s3cr3t");
        ciphertext[ciphertext.length - 1] ^= 0x01;
        assertThatThrownBy(() -> cipher.decrypt(ciphertext)).isInstanceOf(IllegalStateException.class);
    }

    @Test
    void rejectsCiphertextTooShortToHoldAnInitializationVector() {
        assertThatThrownBy(() -> cipher.decrypt(new byte[4])).isInstanceOf(IllegalStateException.class);
    }

    @Test
    void treatsNullAsAbsentSoConnectionsWithoutPasswordsAreSupported() {
        assertThat(cipher.encrypt(null)).isNull();
        assertThat(cipher.decrypt(null)).isNull();
    }

    @Test
    void cannotDecryptWithADifferentKeySoASwappedMasterKeyFailsLoudly() {
        byte[] ciphertext = cipher.encrypt("s3cr3t");
        byte[] otherKey = new byte[32];
        Arrays.fill(otherKey, (byte) 9);
        SecretCipher other = new SecretCipher(purpose -> otherKey);
        assertThatThrownBy(() -> other.decrypt(ciphertext)).isInstanceOf(IllegalStateException.class);
    }
}
