package com.github.controlpanel.repo;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

class RepoServiceNormalizeTest {

    @Test
    void acceptsOwnerSlashName() {
        assertEquals("spring-projects/spring-boot", RepoService.normalize("spring-projects/spring-boot"));
    }

    @Test
    void stripsGithubUrlAndGitSuffixAndTrailingSlash() {
        assertEquals("facebook/react", RepoService.normalize("https://github.com/facebook/react.git"));
        assertEquals("facebook/react", RepoService.normalize("http://github.com/facebook/react/"));
    }

    @Test
    void keepsOnlyOwnerAndNameFromDeepUrls() {
        assertEquals("vercel/next.js", RepoService.normalize("https://github.com/vercel/next.js/tree/canary"));
    }

    @Test
    void rejectsInputWithoutOwnerAndName() {
        assertNull(RepoService.normalize("just-a-name"));
        assertNull(RepoService.normalize(""));
        assertNull(RepoService.normalize(null));
        assertNull(RepoService.normalize("   "));
    }
}
