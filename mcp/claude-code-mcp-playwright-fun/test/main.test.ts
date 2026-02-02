import { assertEquals } from "jsr:@std/assert";
import { greet } from "../src/main.ts";

Deno.test("greet should return hello message with name", () => {
  const result = greet("World");
  assertEquals(result, "Hello, World!");
});

Deno.test("greet should return hello message with custom name", () => {
  const result = greet("TypeScript");
  assertEquals(result, "Hello, TypeScript!");
});
