package com.github.diegopacheco.adminconsole.crypto;

public interface MasterKeyProvider {
    byte[] key(String purpose);
}
