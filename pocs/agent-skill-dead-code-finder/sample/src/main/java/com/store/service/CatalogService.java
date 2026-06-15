package com.store.service;

import com.store.model.Product;
import com.store.util.StringUtils;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class CatalogService {

    public List<Product> list() {
        Product book = new Product(StringUtils.slugify("The Pragmatic Programmer"), "The Pragmatic Programmer", 4200);
        Product mug = new Product(StringUtils.slugify("Coffee Mug"), "Coffee Mug", 1500);
        return List.of(book, mug);
    }

    public void rebuildIndex() {
        StringUtils.slugify("rebuild");
    }
}
