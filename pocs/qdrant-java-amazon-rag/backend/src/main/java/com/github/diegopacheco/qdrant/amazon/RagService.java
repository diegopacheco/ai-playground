package com.github.diegopacheco.qdrant.amazon;

import java.util.List;
import org.springframework.stereotype.Service;

@Service
class RagService {

    static final int SOURCES = 5;
    static final int ABOUT_LIMIT = 600;
    static final String NOTHING_FOUND = "I found nothing suitable in the catalog.";

    private final SearchService search;
    private final OllamaClient ollama;

    RagService(SearchService search, OllamaClient ollama) {
        this.search = search;
        this.ollama = ollama;
    }

    AskResponse ask(String question, String category, Double minPrice, Double maxPrice) {
        List<Hit> sources = search.search(question, SOURCES, category, minPrice, maxPrice);
        if (sources.isEmpty()) {
            return new AskResponse(NOTHING_FOUND, sources);
        }
        return new AskResponse(ollama.chat(prompt(question, sources)), sources);
    }

    static String prompt(String question, List<Hit> sources) {
        StringBuilder prompt = new StringBuilder();
        prompt.append("You are a shopping assistant for a catalog of products sold on Amazon.\n");
        prompt.append("Recommend only products from the numbered list below and mention them by name and price.\n");
        prompt.append("If none of them fits the question, reply exactly: ").append(NOTHING_FOUND).append("\n\n");
        prompt.append("Products:\n");
        for (int i = 0; i < sources.size(); i++) {
            Hit hit = sources.get(i);
            String about = hit.about().length() > ABOUT_LIMIT ? hit.about().substring(0, ABOUT_LIMIT) : hit.about();
            prompt.append(i + 1).append(". ").append(hit.name()).append("\n");
            prompt.append("   Price: ").append(hit.price() == null ? "unknown" : "$" + String.format("%.2f", hit.price())).append("\n");
            prompt.append("   Category: ").append(hit.category()).append("\n");
            prompt.append("   About: ").append(about).append("\n");
        }
        prompt.append("\nQuestion: ").append(question).append("\nAnswer:");
        return prompt.toString();
    }
}
