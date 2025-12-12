package com.github.diegopacheco.embabel.agent;

import com.embabel.agent.api.annotation.Action;
import com.embabel.agent.api.annotation.AchievesGoal;
import com.embabel.agent.api.annotation.Export;
import com.embabel.agent.api.annotation.Agent;
import com.embabel.agent.api.common.OperationContext;
import com.embabel.agent.domain.io.UserInput;

@Agent(description = "Creates blog posts based on user input topics")
public class BlogPostAgent {

    @AchievesGoal(description = "Write a blog post", export = @Export(startingInputTypes = {UserInput.class}))
    @Action
    public BlogPost writeBlogPost(UserInput input, OperationContext context) {
        String prompt = input.getContent();
        if (!prompt.endsWith(".")) {
            prompt = prompt + ".";
        }
        prompt = prompt + " The blog post should have a catchy title, engaging content (at least 500 words), and a brief summary at the end.";
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
