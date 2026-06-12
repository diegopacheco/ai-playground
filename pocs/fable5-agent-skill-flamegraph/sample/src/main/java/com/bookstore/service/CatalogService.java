package com.bookstore.service;

import com.bookstore.model.Book;
import com.bookstore.repository.BookRepository;
import java.util.List;
import org.springframework.stereotype.Service;

@Service
public class CatalogService {

    private final BookRepository bookRepository;

    public CatalogService(BookRepository bookRepository) {
        this.bookRepository = bookRepository;
    }

    public List<Book> listBooks() {
        return bookRepository.findAll();
    }

    public double priceOf(String bookId) {
        Book book = bookRepository.findById(bookId);
        if (book == null) {
            throw new IllegalArgumentException("unknown book " + bookId);
        }
        return book.price();
    }
}
