#!/usr/bin/env python3
import os
import sys
import shutil
import subprocess
from datetime import datetime, timedelta

REPO = ""
COUNT = 0
BASE = datetime.now() - timedelta(days=45)

BAD = [
    "wip", "fix", "stuff", "asdf", "update", "more", "...", ".", "tmp", "changes",
    "fix2", "ok", "final", "final v2", "pls work", "oops", "minor", "test", "save",
    "commit", "aaa", "done", "quick fix", "cleanup", "misc", "x", "fixes",
    "update code", "blah", "work", "wip2", "nit", "stuff again", "hmm",
    "revert maybe", "typo", "yolo", "add things", "later", "fixing",
]

ENTITIES = [
    {"name": "User", "pkg": "user", "plural": "users",
     "fields": [("String", "name"), ("String", "email")]},
    {"name": "Project", "pkg": "project", "plural": "projects",
     "fields": [("String", "name"), ("String", "description")]},
    {"name": "Task", "pkg": "task", "plural": "tasks",
     "fields": [("String", "title"), ("String", "status")]},
    {"name": "Tag", "pkg": "tag", "plural": "tags",
     "fields": [("String", "name"), ("String", "color")]},
    {"name": "Comment", "pkg": "comment", "plural": "comments",
     "fields": [("String", "body"), ("String", "author")]},
    {"name": "Milestone", "pkg": "milestone", "plural": "milestones",
     "fields": [("String", "name"), ("String", "dueDate")]},
]

ROOT_PKG = "com.github.diegopacheco.taskapi"
SRC = "src/main/java/com/github/diegopacheco/taskapi"
TEST = "src/test/java/com/github/diegopacheco/taskapi"


def cap(s):
    return s[0].upper() + s[1:]


def run(args, env=None):
    subprocess.run(["git", "-C", REPO] + args, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)


def write(rel, content):
    full = os.path.join(REPO, rel)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w") as fh:
        fh.write(content if content.endswith("\n") else content + "\n")


def insert(rel, snippet):
    full = os.path.join(REPO, rel)
    with open(full) as fh:
        text = fh.read()
    lines = text.rstrip("\n").split("\n")
    j = len(lines) - 1
    while j >= 0 and lines[j].strip() != "}":
        j -= 1
    lines.insert(j, snippet.rstrip("\n"))
    with open(full, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def commit():
    global COUNT
    msg = BAD[COUNT % len(BAD)]
    when = (BASE + timedelta(hours=10 * COUNT)).strftime("%Y-%m-%dT%H:%M:%S")
    env = dict(os.environ)
    env["GIT_AUTHOR_DATE"] = when
    env["GIT_COMMITTER_DATE"] = when
    run(["add", "-A"])
    run(["commit", "-q", "-m", msg], env=env)
    COUNT += 1


def step(rel, content):
    write(rel, content)
    commit()


def step_insert(rel, snippet):
    insert(rel, snippet)
    commit()


def entity_source(e):
    fields = e["fields"]
    decls = []
    for typ, fn in fields:
        decls.append("    private " + typ + " " + fn + ";")
        decls.append("    public " + typ + " get" + cap(fn) + "() { return " + fn + "; }")
        decls.append("    public void set" + cap(fn) + "(" + typ + " " + fn +
                     ") { this." + fn + " = " + fn + "; }")
    body = "\n".join(decls)
    return ("package " + ROOT_PKG + "." + e["pkg"] + ";\n\n"
            "import " + ROOT_PKG + ".common.BaseEntity;\n"
            "import jakarta.persistence.Entity;\n"
            "import jakarta.persistence.Table;\n\n"
            "@Entity\n"
            "@Table(name = \"" + e["plural"] + "\")\n"
            "public class " + e["name"] + " extends BaseEntity {\n" + body + "\n}\n")


def repo_source(e):
    return ("package " + ROOT_PKG + "." + e["pkg"] + ";\n\n"
            "import java.util.List;\n"
            "import java.util.Optional;\n"
            "import org.springframework.data.jpa.repository.JpaRepository;\n\n"
            "public interface " + e["name"] + "Repository extends JpaRepository<" +
            e["name"] + ", Long> {\n}\n")


def dto_source(e):
    comps = ["Long id"]
    for typ, fn in e["fields"]:
        comps.append(typ + " " + fn)
    return ("package " + ROOT_PKG + "." + e["pkg"] + ";\n\n"
            "public record " + e["name"] + "Dto(" + ", ".join(comps) + ") {\n}\n")


def service_source(e):
    name = e["name"]
    args = ", ".join(["e.get" + cap(fn) + "()" for _, fn in e["fields"]])
    apply_lines = "\n".join(
        ["        e.set" + cap(fn) + "(dto." + fn + "());" for _, fn in e["fields"]])
    return ("package " + ROOT_PKG + "." + e["pkg"] + ";\n\n"
            "import java.util.List;\n"
            "import org.springframework.stereotype.Service;\n"
            "import " + ROOT_PKG + ".common.ResourceNotFoundException;\n\n"
            "@Service\n"
            "public class " + name + "Service {\n\n"
            "    private final " + name + "Repository repo;\n\n"
            "    public " + name + "Service(" + name + "Repository repo) {\n"
            "        this.repo = repo;\n"
            "    }\n\n"
            "    public List<" + name + "Dto> findAll() {\n"
            "        return repo.findAll().stream().map(this::toDto).toList();\n"
            "    }\n\n"
            "    public " + name + "Dto findById(Long id) {\n"
            "        var e = repo.findById(id).orElseThrow(() -> new ResourceNotFoundException(\"" +
            name + "\", id));\n"
            "        return toDto(e);\n"
            "    }\n\n"
            "    public " + name + "Dto create(" + name + "Dto dto) {\n"
            "        var e = new " + name + "();\n"
            "        apply(e, dto);\n"
            "        return toDto(repo.save(e));\n"
            "    }\n\n"
            "    private " + name + "Dto toDto(" + name + " e) {\n"
            "        return new " + name + "Dto(e.getId(), " + args + ");\n"
            "    }\n\n"
            "    private void apply(" + name + " e, " + name + "Dto dto) {\n" +
            apply_lines + "\n"
            "    }\n}\n")


def controller_source(e):
    name = e["name"]
    return ("package " + ROOT_PKG + "." + e["pkg"] + ";\n\n"
            "import java.util.List;\n"
            "import org.springframework.http.HttpStatus;\n"
            "import org.springframework.web.bind.annotation.*;\n\n"
            "@RestController\n"
            "@RequestMapping(\"/api/" + e["plural"] + "\")\n"
            "public class " + name + "Controller {\n\n"
            "    private final " + name + "Service service;\n\n"
            "    public " + name + "Controller(" + name + "Service service) {\n"
            "        this.service = service;\n"
            "    }\n\n"
            "    @GetMapping\n"
            "    public List<" + name + "Dto> all() {\n"
            "        return service.findAll();\n"
            "    }\n\n"
            "    @GetMapping(\"/{id}\")\n"
            "    public " + name + "Dto byId(@PathVariable Long id) {\n"
            "        return service.findById(id);\n"
            "    }\n\n"
            "    @PostMapping\n"
            "    @ResponseStatus(HttpStatus.CREATED)\n"
            "    public " + name + "Dto create(@RequestBody " + name + "Dto dto) {\n"
            "        return service.create(dto);\n"
            "    }\n}\n")


def service_update(e):
    name = e["name"]
    return ("\n    public " + name + "Dto update(Long id, " + name + "Dto dto) {\n"
            "        var e = repo.findById(id).orElseThrow(() -> new ResourceNotFoundException(\"" +
            name + "\", id));\n"
            "        apply(e, dto);\n"
            "        return toDto(repo.save(e));\n"
            "    }\n")


def service_delete(e):
    name = e["name"]
    return ("\n    public void delete(Long id) {\n"
            "        if (!repo.existsById(id)) {\n"
            "            throw new ResourceNotFoundException(\"" + name + "\", id);\n"
            "        }\n"
            "        repo.deleteById(id);\n"
            "    }\n")


def controller_update(e):
    name = e["name"]
    return ("\n    @PutMapping(\"/{id}\")\n"
            "    public " + name + "Dto update(@PathVariable Long id, @RequestBody " +
            name + "Dto dto) {\n"
            "        return service.update(id, dto);\n"
            "    }\n")


def controller_delete(e):
    return ("\n    @DeleteMapping(\"/{id}\")\n"
            "    @ResponseStatus(HttpStatus.NO_CONTENT)\n"
            "    public void delete(@PathVariable Long id) {\n"
            "        service.delete(id);\n"
            "    }\n")


def bootstrap():
    step("pom.xml", POM)
    step(".gitignore", GITIGNORE)
    step("src/main/resources/application.yml", APP_YML_V1)
    step(SRC + "/TaskApiApplication.java", APPLICATION)
    step("README.md", README_V1)
    step("src/main/resources/banner.txt", BANNER)


def common():
    step(SRC + "/common/BaseEntity.java", BASE_ENTITY)
    step(SRC + "/common/ResourceNotFoundException.java", RNFE_V1)
    step(SRC + "/common/ApiError.java", API_ERROR)
    step(SRC + "/common/GlobalExceptionHandler.java", GEH)
    step_insert(SRC + "/common/GlobalExceptionHandler.java", GEH_BAD_REQUEST)
    step(SRC + "/common/ApiResponse.java", API_RESPONSE)
    step(SRC + "/common/PageResponse.java", PAGE_RESPONSE)


def config():
    step(SRC + "/config/WebConfig.java", WEB_CONFIG)
    step(SRC + "/config/AppConfig.java", APP_CONFIG)


def entities():
    for e in ENTITIES:
        base = SRC + "/" + e["pkg"] + "/"
        step(base + e["name"] + ".java", entity_source(e))
        step(base + e["name"] + "Repository.java", repo_source(e))
        step(base + e["name"] + "Dto.java", dto_source(e))
        step(base + e["name"] + "Service.java", service_source(e))
        step(base + e["name"] + "Controller.java", controller_source(e))
        step_insert(base + e["name"] + "Service.java", service_update(e))
        step_insert(base + e["name"] + "Controller.java", controller_update(e))
        step_insert(base + e["name"] + "Service.java", service_delete(e))
        step_insert(base + e["name"] + "Controller.java", controller_delete(e))


def incremental():
    step_insert(SRC + "/task/Task.java",
                "    private String priority;\n"
                "    public String getPriority() { return priority; }\n"
                "    public void setPriority(String priority) { this.priority = priority; }\n")
    step_insert(SRC + "/task/TaskRepository.java",
                "    List<Task> findByStatus(String status);\n")
    step_insert(SRC + "/task/TaskService.java",
                "\n    public List<TaskDto> findByStatus(String status) {\n"
                "        return repo.findByStatus(status).stream().map(this::toDto).toList();\n"
                "    }\n")
    step_insert(SRC + "/task/TaskController.java",
                "\n    @GetMapping(\"/by-status\")\n"
                "    public List<TaskDto> byStatus(@RequestParam String status) {\n"
                "        return service.findByStatus(status);\n"
                "    }\n")
    step_insert(SRC + "/project/ProjectRepository.java",
                "    List<Project> findByNameContainingIgnoreCase(String q);\n")
    step_insert(SRC + "/project/ProjectService.java",
                "\n    public List<ProjectDto> search(String q) {\n"
                "        return repo.findByNameContainingIgnoreCase(q).stream().map(this::toDto).toList();\n"
                "    }\n")
    step_insert(SRC + "/project/ProjectController.java",
                "\n    @GetMapping(\"/search\")\n"
                "    public List<ProjectDto> search(@RequestParam String q) {\n"
                "        return service.search(q);\n"
                "    }\n")
    step_insert(SRC + "/comment/CommentRepository.java",
                "    List<Comment> findByAuthor(String author);\n")
    step_insert(SRC + "/comment/CommentService.java",
                "\n    public List<CommentDto> findByAuthor(String author) {\n"
                "        return repo.findByAuthor(author).stream().map(this::toDto).toList();\n"
                "    }\n")
    step_insert(SRC + "/comment/CommentController.java",
                "\n    @GetMapping(\"/by-author\")\n"
                "    public List<CommentDto> byAuthor(@RequestParam String author) {\n"
                "        return service.findByAuthor(author);\n"
                "    }\n")
    step_insert(SRC + "/user/UserRepository.java",
                "    Optional<User> findByEmail(String email);\n")
    step_insert(SRC + "/user/UserService.java",
                "\n    public UserDto findByEmail(String email) {\n"
                "        return repo.findByEmail(email).map(this::toDto)\n"
                "                .orElseThrow(() -> new ResourceNotFoundException(\"User\", 0L));\n"
                "    }\n")
    step_insert(SRC + "/user/UserController.java",
                "\n    @GetMapping(\"/by-email\")\n"
                "    public UserDto byEmail(@RequestParam String email) {\n"
                "        return service.byEmail(email);\n"
                "    }\n".replace("service.byEmail(email)", "service.findByEmail(email)"))
    step(SRC + "/common/ResourceNotFoundException.java", RNFE_V2)
    step_insert(SRC + "/tag/TagService.java",
                "\n    public long count() {\n"
                "        return repo.count();\n"
                "    }\n")
    step_insert(SRC + "/tag/TagController.java",
                "\n    @GetMapping(\"/count\")\n"
                "    public long count() {\n"
                "        return service.count();\n"
                "    }\n")
    step_insert(SRC + "/milestone/MilestoneService.java",
                "\n    public List<MilestoneDto> upcoming() {\n"
                "        return findAll();\n"
                "    }\n")
    step_insert(SRC + "/milestone/MilestoneController.java",
                "\n    @GetMapping(\"/upcoming\")\n"
                "    public List<MilestoneDto> upcoming() {\n"
                "        return service.upcoming();\n"
                "    }\n")
    step("src/main/resources/application.yml", APP_YML_V2)
    step("README.md", README_V2)


def tests():
    step(TEST + "/TaskApiApplicationTests.java", TEST_CONTEXT)
    step(TEST + "/common/ApiResponseTest.java", TEST_API_RESPONSE)
    step(TEST + "/common/PageResponseTest.java", TEST_PAGE_RESPONSE)
    step(TEST + "/common/BaseEntityTest.java", TEST_BASE_ENTITY)
    step(TEST + "/task/TaskTest.java", TEST_TASK)
    step(TEST + "/user/UserDtoTest.java", TEST_USER_DTO)
    step(TEST + "/project/ProjectDtoTest.java", TEST_PROJECT_DTO)
    step(TEST + "/common/ResourceNotFoundExceptionTest.java", TEST_RNFE)
    step(TEST + "/common/ApiErrorTest.java", TEST_API_ERROR)
    step(TEST + "/tag/TagTest.java", TEST_TAG)
    step(TEST + "/comment/CommentTest.java", TEST_COMMENT)


def main():
    global REPO
    target = sys.argv[1] if len(sys.argv) > 1 else "/tmp/task-api-bad-history"
    REPO = os.path.abspath(target)
    if os.path.exists(REPO):
        shutil.rmtree(REPO)
    os.makedirs(REPO)
    run(["init", "-q"])
    run(["config", "user.name", "taskapi-dev"])
    run(["config", "user.email", "dev@taskapi.local"])
    run(["config", "commit.gpgsign", "false"])
    bootstrap()
    common()
    config()
    entities()
    incremental()
    tests()
    print("generated " + str(COUNT) + " commits at " + REPO)


POM = """<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
  <modelVersion>4.0.0</modelVersion>
  <parent>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-parent</artifactId>
    <version>4.0.6</version>
  </parent>
  <groupId>com.github.diegopacheco.taskapi</groupId>
  <artifactId>task-api</artifactId>
  <version>1.0-SNAPSHOT</version>
  <packaging>jar</packaging>
  <properties>
    <java.version>25</java.version>
    <maven.compiler.source>25</maven.compiler.source>
    <maven.compiler.target>25</maven.compiler.target>
  </properties>
  <dependencies>
    <dependency>
      <groupId>org.springframework.boot</groupId>
      <artifactId>spring-boot-starter-web</artifactId>
    </dependency>
    <dependency>
      <groupId>org.springframework.boot</groupId>
      <artifactId>spring-boot-starter-data-jpa</artifactId>
    </dependency>
    <dependency>
      <groupId>com.h2database</groupId>
      <artifactId>h2</artifactId>
      <scope>runtime</scope>
    </dependency>
    <dependency>
      <groupId>org.springframework.boot</groupId>
      <artifactId>spring-boot-starter-test</artifactId>
      <scope>test</scope>
      <exclusions>
        <exclusion>
          <groupId>org.junit.vintage</groupId>
          <artifactId>junit-vintage-engine</artifactId>
        </exclusion>
      </exclusions>
    </dependency>
  </dependencies>
  <build>
    <plugins>
      <plugin>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-maven-plugin</artifactId>
      </plugin>
    </plugins>
  </build>
</project>
"""

GITIGNORE = """target/
*.class
.idea/
*.iml
.DS_Store
"""

APP_YML_V1 = """spring:
  application:
    name: task-api
"""

APP_YML_V2 = """spring:
  application:
    name: task-api
  jpa:
    hibernate:
      ddl-auto: create-drop
    open-in-view: false
server:
  port: 8080
"""

APPLICATION = """package com.github.diegopacheco.taskapi;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class TaskApiApplication {

    public static void main(String[] args) {
        SpringApplication.run(TaskApiApplication.class, args);
    }
}
"""

README_V1 = """# task-api

A service.
"""

README_V2 = """# task-api

Task management REST API built with Spring Boot 4 and Java 25.

Entities: User, Project, Task, Tag, Comment, Milestone.
Each exposes CRUD endpoints under /api.

Run: mvn spring-boot:run
Test: mvn test
"""

BANNER = """task-api
"""

BASE_ENTITY = """package com.github.diegopacheco.taskapi.common;

import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.MappedSuperclass;
import java.time.Instant;

@MappedSuperclass
public abstract class BaseEntity {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private Instant createdAt = Instant.now();

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(Instant createdAt) {
        this.createdAt = createdAt;
    }
}
"""

RNFE_V1 = """package com.github.diegopacheco.taskapi.common;

public class ResourceNotFoundException extends RuntimeException {

    public ResourceNotFoundException(String entity, Long id) {
        super(entity + " " + id + " missing");
    }
}
"""

RNFE_V2 = """package com.github.diegopacheco.taskapi.common;

public class ResourceNotFoundException extends RuntimeException {

    public ResourceNotFoundException(String entity, Long id) {
        super(entity + " not found: " + id);
    }
}
"""

API_ERROR = """package com.github.diegopacheco.taskapi.common;

import java.time.Instant;

public record ApiError(Instant timestamp, int status, String error, String message) {
}
"""

GEH = """package com.github.diegopacheco.taskapi.common;

import java.time.Instant;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

@RestControllerAdvice
public class GlobalExceptionHandler {

    @ExceptionHandler(ResourceNotFoundException.class)
    public ResponseEntity<ApiError> notFound(ResourceNotFoundException ex) {
        var body = new ApiError(Instant.now(), 404, "Not Found", ex.getMessage());
        return ResponseEntity.status(HttpStatus.NOT_FOUND).body(body);
    }
}
"""

GEH_BAD_REQUEST = """
    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<ApiError> badRequest(IllegalArgumentException ex) {
        var body = new ApiError(Instant.now(), 400, "Bad Request", ex.getMessage());
        return ResponseEntity.badRequest().body(body);
    }
"""

API_RESPONSE = """package com.github.diegopacheco.taskapi.common;

public record ApiResponse<T>(boolean success, T data, String message) {

    public static <T> ApiResponse<T> ok(T data) {
        return new ApiResponse<>(true, data, "ok");
    }
}
"""

PAGE_RESPONSE = """package com.github.diegopacheco.taskapi.common;

import java.util.List;

public record PageResponse<T>(List<T> items, int page, int size, long total) {
}
"""

WEB_CONFIG = """package com.github.diegopacheco.taskapi.config;

import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
public class WebConfig implements WebMvcConfigurer {

    @Override
    public void addCorsMappings(CorsRegistry registry) {
        registry.addMapping("/api/**")
                .allowedOrigins("*")
                .allowedMethods("GET", "POST", "PUT", "DELETE");
    }
}
"""

APP_CONFIG = """package com.github.diegopacheco.taskapi.config;

import java.time.Clock;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class AppConfig {

    @Bean
    public Clock clock() {
        return Clock.systemUTC();
    }
}
"""

TEST_CONTEXT = """package com.github.diegopacheco.taskapi;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
class TaskApiApplicationTests {

    @Test
    void contextLoads() {
    }
}
"""

TEST_API_RESPONSE = """package com.github.diegopacheco.taskapi.common;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class ApiResponseTest {

    @Test
    void okWrapsData() {
        var r = ApiResponse.ok("hello");
        assertTrue(r.success());
        assertEquals("hello", r.data());
    }
}
"""

TEST_PAGE_RESPONSE = """package com.github.diegopacheco.taskapi.common;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import org.junit.jupiter.api.Test;

class PageResponseTest {

    @Test
    void holdsPaging() {
        var p = new PageResponse<>(List.of("a", "b"), 0, 2, 2L);
        assertEquals(2, p.items().size());
        assertEquals(2L, p.total());
    }
}
"""

TEST_BASE_ENTITY = """package com.github.diegopacheco.taskapi.common;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import com.github.diegopacheco.taskapi.user.User;
import org.junit.jupiter.api.Test;

class BaseEntityTest {

    @Test
    void idRoundTrips() {
        var u = new User();
        u.setId(7L);
        assertEquals(7L, u.getId());
        assertNotNull(u.getCreatedAt());
    }
}
"""

TEST_TASK = """package com.github.diegopacheco.taskapi.task;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class TaskTest {

    @Test
    void fields() {
        var t = new Task();
        t.setTitle("ship it");
        t.setStatus("open");
        t.setPriority("high");
        assertEquals("open", t.getStatus());
        assertEquals("high", t.getPriority());
    }
}
"""

TEST_USER_DTO = """package com.github.diegopacheco.taskapi.user;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class UserDtoTest {

    @Test
    void accessors() {
        var d = new UserDto(1L, "ann", "ann@taskapi.local");
        assertEquals("ann", d.name());
        assertEquals("ann@taskapi.local", d.email());
    }
}
"""

TEST_PROJECT_DTO = """package com.github.diegopacheco.taskapi.project;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class ProjectDtoTest {

    @Test
    void accessors() {
        var d = new ProjectDto(2L, "apollo", "moon");
        assertEquals("apollo", d.name());
        assertEquals("moon", d.description());
    }
}
"""

TEST_RNFE = """package com.github.diegopacheco.taskapi.common;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class ResourceNotFoundExceptionTest {

    @Test
    void message() {
        var e = new ResourceNotFoundException("Task", 5L);
        assertTrue(e.getMessage().contains("Task"));
        assertTrue(e.getMessage().contains("5"));
    }
}
"""

TEST_API_ERROR = """package com.github.diegopacheco.taskapi.common;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.time.Instant;
import org.junit.jupiter.api.Test;

class ApiErrorTest {

    @Test
    void holds() {
        var e = new ApiError(Instant.now(), 404, "Not Found", "nope");
        assertEquals(404, e.status());
    }
}
"""

TEST_TAG = """package com.github.diegopacheco.taskapi.tag;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class TagTest {

    @Test
    void fields() {
        var t = new Tag();
        t.setName("urgent");
        t.setColor("red");
        assertEquals("urgent", t.getName());
        assertEquals("red", t.getColor());
    }
}
"""

TEST_COMMENT = """package com.github.diegopacheco.taskapi.comment;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class CommentTest {

    @Test
    void fields() {
        var c = new Comment();
        c.setBody("looks good");
        c.setAuthor("ann");
        assertEquals("looks good", c.getBody());
        assertEquals("ann", c.getAuthor());
    }
}
"""

if __name__ == "__main__":
    main()
