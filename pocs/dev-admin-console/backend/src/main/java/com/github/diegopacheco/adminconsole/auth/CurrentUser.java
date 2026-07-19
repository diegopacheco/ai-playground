package com.github.diegopacheco.adminconsole.auth;

import jakarta.servlet.http.HttpServletRequest;
import org.springframework.stereotype.Component;
import org.springframework.web.context.request.RequestContextHolder;
import org.springframework.web.context.request.ServletRequestAttributes;

@Component
public class CurrentUser {
    public static final String ATTRIBUTE = "adminConsolePrincipal";

    public Jwt.Principal principal() {
        var attributes = RequestContextHolder.getRequestAttributes();
        if (attributes instanceof ServletRequestAttributes servlet) {
            HttpServletRequest request = servlet.getRequest();
            if (request.getAttribute(ATTRIBUTE) instanceof Jwt.Principal principal) {
                return principal;
            }
        }
        return null;
    }

    public String username() {
        Jwt.Principal principal = principal();
        return principal == null ? null : principal.username();
    }

    public boolean isAdmin() {
        Jwt.Principal principal = principal();
        return principal != null && "admin".equals(principal.role());
    }
}
