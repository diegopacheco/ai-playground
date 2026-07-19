package com.github.diegopacheco.devadminconsole.crypto;

public interface MasterKeyProvider {
    byte[] key(String purpose);
}
