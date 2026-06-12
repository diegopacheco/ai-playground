package com.bookstore.api;

import com.bookstore.model.Book;
import com.bookstore.model.Receipt;
import com.bookstore.service.CatalogService;
import com.bookstore.service.CheckoutService;
import java.util.List;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class CheckoutController {

    private final CheckoutService checkoutService;
    private final CatalogService catalogService;

    public CheckoutController(CheckoutService checkoutService, CatalogService catalogService) {
        this.checkoutService = checkoutService;
        this.catalogService = catalogService;
    }

    @GetMapping("/catalog")
    public List<Book> catalog() {
        return catalogService.listBooks();
    }

    @PostMapping("/checkout")
    public Receipt checkout(@RequestParam String customerId,
                            @RequestParam(defaultValue = "CA") String region) {
        return checkoutService.checkout(customerId, region);
    }
}
