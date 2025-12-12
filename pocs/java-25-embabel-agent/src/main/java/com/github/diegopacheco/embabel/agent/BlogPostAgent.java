package com.github.diegopacheco.embabel.agent;

import com.embabel.agent.api.annotation.Action;
import com.embabel.agent.api.annotation.AchievesGoal;
import com.embabel.agent.api.annotation.Agent;
import com.embabel.agent.api.common.OperationContext;

@Agent(description = "Creates blog posts based on user input topics")
public class BlogPostAgent {

    @Action
    public BlogPost writeBlogPost(BlogInput input, OperationContext context) {
        String prompt = String.format(
            "Write a blog post about: %s. " +
            "Keywords to include: %s. " +
            "Tone should be: %s. " +
            "The blog post should have a catchy title, engaging content (at least 500 words), " +
            "and a brief summary at the end.",
            input.topic(),
            input.keywords().isEmpty() ? "relevant to the topic" : input.keywords(),
            input.tone()
        );
        return context.ai()
            .withDefaultLlm()
            .createObject(prompt, BlogPost.class);
    }

    @AchievesGoal(description = "Review and finalize the blog post")
    @Action
    public ReviewedBlogPost reviewBlogPost(BlogPost post, OperationContext context) {
        String prompt = String.format(
            "Review this blog post and provide feedback. " +
            "Title: %s. " +
            "Content: %s. " +
            "Summary: %s. " +
            "Provide constructive feedback, suggest improvements if needed, " +
            "and rate the quality from 1 to 10.",
            post.title(),
            post.content(),
            post.summary()
        );
        return context.ai()
            .withDefaultLlm()
            .createObject(prompt, ReviewedBlogPost.class);
    }
}
