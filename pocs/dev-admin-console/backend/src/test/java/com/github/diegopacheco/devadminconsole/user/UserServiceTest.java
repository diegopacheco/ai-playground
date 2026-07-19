package com.github.diegopacheco.devadminconsole.user;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.github.diegopacheco.devadminconsole.auth.PasswordHasher;
import java.time.Instant;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

class UserServiceTest {
    private UserRepository repository;
    private UserService service;
    private final PasswordHasher hasher = new PasswordHasher();

    @BeforeEach
    void setUp() {
        repository = Mockito.mock(UserRepository.class);
        service = new UserService(repository, hasher);
    }

    private User user(long id, String username, String role, boolean enabled, String password) {
        byte[] salt = hasher.salt();
        return new User(id, username, hasher.hash(password, salt), salt, role, enabled, Instant.now(), null);
    }

    @Test
    void authenticatesAValidUserAndRecordsTheLoginForTheAuditTrail() {
        User stored = user(1, "diego", User.ADMIN, true, "secret");
        when(repository.findByUsername("diego")).thenReturn(Optional.of(stored));
        assertThat(service.authenticate("diego", "secret")).isPresent();
        verify(repository).touchLogin(1L);
    }

    @Test
    void rejectsTheWrongPasswordWithoutRecordingALogin() {
        when(repository.findByUsername("diego")).thenReturn(Optional.of(user(1, "diego", User.ADMIN, true, "secret")));
        assertThat(service.authenticate("diego", "wrong")).isEmpty();
        verify(repository, never()).touchLogin(anyLong());
    }

    @Test
    void refusesADisabledUserEvenWithTheCorrectPasswordSoRevokingAccessActuallyWorks() {
        when(repository.findByUsername("diego")).thenReturn(Optional.of(user(1, "diego", User.ADMIN, false, "secret")));
        assertThat(service.authenticate("diego", "secret")).isEmpty();
    }

    @Test
    void rejectsAnUnknownUserWithoutRevealingThatTheNameDoesNotExist() {
        when(repository.findByUsername("ghost")).thenReturn(Optional.empty());
        assertThat(service.authenticate("ghost", "secret")).isEmpty();
    }

    @Test
    void refusesToDeleteTheLastAdminSoTheConsoleCannotBeLockedOutOfItsOwnConfiguration() {
        User onlyAdmin = user(1, "admin", User.ADMIN, true, "admin");
        when(repository.findById(1L)).thenReturn(Optional.of(onlyAdmin));
        when(repository.findAll()).thenReturn(List.of(onlyAdmin, user(2, "reader", User.USER, true, "x")));
        assertThatThrownBy(() -> service.delete(1)).hasMessageContaining("last admin");
        verify(repository, never()).delete(anyLong());
    }

    @Test
    void refusesToDemoteTheLastAdminForTheSameReason() {
        User onlyAdmin = user(1, "admin", User.ADMIN, true, "admin");
        when(repository.findById(1L)).thenReturn(Optional.of(onlyAdmin));
        when(repository.findAll()).thenReturn(List.of(onlyAdmin));
        assertThatThrownBy(() -> service.updateRoleAndEnabled(1, User.USER, true)).hasMessageContaining("last admin");
    }

    @Test
    void allowsDeletingAnAdminWhenAnotherEnabledAdminRemains() {
        User first = user(1, "admin", User.ADMIN, true, "admin");
        User second = user(2, "diego", User.ADMIN, true, "x");
        when(repository.findById(1L)).thenReturn(Optional.of(first));
        when(repository.findAll()).thenReturn(List.of(first, second));
        service.delete(1);
        verify(repository).delete(1L);
    }

    @Test
    void doesNotCountDisabledAdminsAsRemainingCoverSinceTheyCannotLogIn() {
        User active = user(1, "admin", User.ADMIN, true, "admin");
        User disabled = user(2, "old", User.ADMIN, false, "x");
        when(repository.findById(1L)).thenReturn(Optional.of(active));
        when(repository.findAll()).thenReturn(List.of(active, disabled));
        assertThatThrownBy(() -> service.delete(1)).hasMessageContaining("last admin");
    }

    @Test
    void rejectsDuplicateUsernames() {
        when(repository.findByUsername("diego")).thenReturn(Optional.of(user(1, "diego", User.USER, true, "x")));
        assertThatThrownBy(() -> service.create("diego", "secret", User.USER)).hasMessageContaining("already exists");
    }

    @Test
    void rejectsAnUnknownRoleSoPrivilegesCannotBeInventedThroughTheApi() {
        when(repository.findByUsername("new")).thenReturn(Optional.empty());
        assertThatThrownBy(() -> service.create("new", "secret", "superadmin")).hasMessageContaining("role must be");
    }

    @Test
    void rejectsBlankUsernamesAndShortPasswords() {
        assertThatThrownBy(() -> service.create("", "secret", User.USER)).hasMessageContaining("username");
        assertThatThrownBy(() -> service.create("new", "ab", User.USER)).hasMessageContaining("password");
    }

    @Test
    void storesADifferentSaltAndHashWhenAPasswordIsChanged() {
        service.changePassword(1, "brand-new");
        verify(repository).updatePassword(anyLong(), any(), any());
    }

    @Test
    void detectsTheUntouchedBootstrapPasswordSoTheUiCanWarnAboutIt() {
        when(repository.findByUsername("admin")).thenReturn(Optional.of(user(1, "admin", User.ADMIN, true, "admin")));
        assertThat(service.usingBootstrapPassword("admin", "admin")).isTrue();
    }

    @Test
    void reportsTheBootstrapPasswordAsChangedOnceItIsDifferent() {
        when(repository.findByUsername("admin")).thenReturn(Optional.of(user(1, "admin", User.ADMIN, true, "rotated")));
        assertThat(service.usingBootstrapPassword("admin", "admin")).isFalse();
    }
}
