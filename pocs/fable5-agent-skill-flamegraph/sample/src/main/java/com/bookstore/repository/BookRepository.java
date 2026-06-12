package com.bookstore.repository;

import com.bookstore.model.Book;
import java.util.List;
import java.util.Map;
import org.springframework.stereotype.Repository;

@Repository
public class BookRepository {

    private final Map<String, Book> books = Map.of(
            "b1", new Book("b1", "The Pragmatic Programmer", "Hunt & Thomas", 42.0),
            "b2", new Book("b2", "Designing Data-Intensive Applications", "Kleppmann", 55.0),
            "b3", new Book("b3", "Domain-Driven Design", "Evans", 61.0),
            "b4", new Book("b4", "Release It!", "Nygard", 38.0));

    public List<Book> findAll() {
        return List.copyOf(books.values());
    }

    public Book findById(String bookId) {
        return books.get(bookId);
    }
}
