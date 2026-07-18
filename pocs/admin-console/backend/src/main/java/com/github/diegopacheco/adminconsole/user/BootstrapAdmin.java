package com.github.diegopacheco.adminconsole.user;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.ApplicationRunner;
import org.springframework.boot.ApplicationArguments;
import org.springframework.stereotype.Component;

@Component
public class BootstrapAdmin implements ApplicationRunner {
    private final UserRepository users;
    private final UserService service;
    private final String username;
    private final String password;

    public BootstrapAdmin(UserRepository users, UserService service,
                          @Value("${app.admin.bootstrap-username}") String username,
                          @Value("${app.admin.bootstrap-password}") String password) {
        this.users = users;
        this.service = service;
        this.username = username;
        this.password = password;
    }

    @Override
    public void run(ApplicationArguments arguments) {
        if (users.count() == 0) {
            service.create(username, password, User.ADMIN);
        }
    }
}
