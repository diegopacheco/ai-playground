package com.github.diegopacheco.qdrant.amazon;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import org.junit.jupiter.api.Test;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;

class ApiControllerTest {

    private static final String UNREACHABLE = "http://localhost:1";

    private final MockMvc mvc = MockMvcBuilders.standaloneSetup(controller()).build();

    private static ApiController controller() {
        QdrantClient qdrant = new QdrantClient(UNREACHABLE);
        OllamaClient ollama = new OllamaClient(UNREACHABLE, "nomic-embed-text", "llama3.2");
        SearchService search = new SearchService(ollama, qdrant);
        return new ApiController(qdrant, search, new RagService(search, ollama), new IngestService(new DatasetClient(), ollama, qdrant), 1000);
    }

    private org.springframework.test.web.servlet.ResultActions postJson(String path, String body) throws Exception {
        return mvc.perform(post(path).contentType(MediaType.APPLICATION_JSON).content(body));
    }

    @Test
    void blankSearchQueryIsRejectedBeforeAnyEmbeddingIsRequested() throws Exception {
        postJson("/api/search", "{\"query\":\"   \"}")
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error").value("query must not be blank"));
    }

    @Test
    void missingSearchQueryIsRejected() throws Exception {
        postJson("/api/search", "{\"category\":\"Toys & Games\"}")
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error").value("query must not be blank"));
    }

    @Test
    void blankQuestionIsRejectedBeforeTheModelIsCalled() throws Exception {
        postJson("/api/ask", "{\"question\":\"\"}")
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error").value("question must not be blank"));
    }

    @Test
    void unreachableOllamaSurfacesAsBadGatewayNotAsAServerBug() throws Exception {
        postJson("/api/search", "{\"query\":\"kite\"}")
                .andExpect(status().isBadGateway())
                .andExpect(jsonPath("$.error").isNotEmpty());
    }

    @Test
    void unreachableQdrantSurfacesAsBadGateway() throws Exception {
        mvc.perform(get("/api/stats"))
                .andExpect(status().isBadGateway())
                .andExpect(jsonPath("$.error").isNotEmpty());
    }
}
