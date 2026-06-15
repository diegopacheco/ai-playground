package com.store.api;

import com.store.model.Order;
import com.store.model.OrderLine;
import com.store.model.Product;
import com.store.service.CatalogService;
import com.store.service.OrderService;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/api")
public class OrderController {

    private final OrderService orderService;
    private final CatalogService catalogService;

    public OrderController(OrderService orderService, CatalogService catalogService) {
        this.orderService = orderService;
        this.catalogService = catalogService;
    }

    @PostMapping("/orders")
    public Order placeOrder(@RequestBody List<OrderLine> lines) {
        return orderService.place(lines);
    }

    @GetMapping("/orders/{id}")
    public Order getOrder(@PathVariable String id) {
        return orderService.find(id);
    }

    @GetMapping("/catalog")
    public List<Product> catalog() {
        return catalogService.list();
    }
}
