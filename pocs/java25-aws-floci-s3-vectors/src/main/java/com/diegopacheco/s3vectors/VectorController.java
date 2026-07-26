package com.diegopacheco.s3vectors;

import io.swagger.v3.oas.annotations.Operation;
import jakarta.validation.Valid;
import java.util.List;
import java.util.Map;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/v1")
public class VectorController {

    private final VectorService service;

    public VectorController(VectorService service) {
        this.service = service;
    }

    @GetMapping("/status")
    @Operation(summary = "Check the application and S3 Vectors infrastructure")
    public Map<String, Object> status() {
        service.ensureInfrastructure();
        return Map.of(
                "status", "UP",
                "bucket", service.bucketName(),
                "index", service.indexName(),
                "dimension", service.dimension());
    }

    @PutMapping("/vectors/{key}")
    @ResponseStatus(HttpStatus.CREATED)
    @Operation(summary = "Create or replace a vector")
    public VectorResponse put(@PathVariable String key, @Valid @RequestBody VectorRequest request) {
        return service.put(key, request.values());
    }

    @GetMapping("/vectors/{key}")
    @Operation(summary = "Get a vector")
    public VectorResponse get(@PathVariable String key) {
        return service.get(key);
    }

    @PostMapping("/vectors/search")
    @Operation(summary = "Search for the nearest vectors")
    public List<SearchResponse> search(@Valid @RequestBody SearchRequest request) {
        return service.search(request.values(), request.topK());
    }

    @DeleteMapping("/vectors/{key}")
    @ResponseStatus(HttpStatus.NO_CONTENT)
    @Operation(summary = "Delete a vector")
    public void delete(@PathVariable String key) {
        service.delete(key);
    }
}
