package com.github.diegopacheco.embabel.agent;

public record ReviewedBlogPost(String title, String content, String summary, String feedback, int qualityScore) {}
