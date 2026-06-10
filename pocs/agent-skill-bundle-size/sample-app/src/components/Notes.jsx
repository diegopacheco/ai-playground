import React from "react";
import { marked } from "marked";
import { notes } from "../data.js";

export default function Notes() {
  const html = marked.parse(notes);
  return (
    <section className="notes">
      <h2>Release notes</h2>
      <div className="md" dangerouslySetInnerHTML={{ __html: html }} />
    </section>
  );
}
