package com.loan.graphql;

import com.loan.domain.AutoLoanDecision;
import com.loan.domain.AutoLoanInput;
import com.loan.service.LoanService;
import org.springframework.graphql.data.method.annotation.Argument;
import org.springframework.graphql.data.method.annotation.MutationMapping;
import org.springframework.graphql.data.method.annotation.QueryMapping;
import org.springframework.stereotype.Controller;

@Controller
public class LoanResolver {

    private final LoanService service;

    public LoanResolver(LoanService service) {
        this.service = service;
    }

    @QueryMapping
    public String health() {
        return "ok";
    }

    @MutationMapping
    public AutoLoanDecision requestAutoLoan(@Argument AutoLoanInput input) {
        return service.evaluate(input);
    }
}
