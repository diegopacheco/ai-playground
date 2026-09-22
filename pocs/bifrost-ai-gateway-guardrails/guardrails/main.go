package main

import (
	"fmt"
	"log"
	"strings"

	"github.com/maximhq/bifrost/core/schemas"
)

func GetName() string {
	return "guardrails"
}

func Cleanup() error {
	return nil
}

func PreLLMHook(ctx *schemas.BifrostContext, req *schemas.BifrostRequest) (*schemas.BifrostRequest, *schemas.LLMPluginShortCircuit, error) {
	text := requestText(req)
	if found := FindPII(text); len(found) > 0 {
		return req, block("pii", found), nil
	}
	if found := FindInjection(text); len(found) > 0 {
		return req, block("prompt_injection", found), nil
	}
	return req, nil, nil
}

func PostLLMHook(ctx *schemas.BifrostContext, resp *schemas.BifrostResponse, bifrostErr *schemas.BifrostError) (*schemas.BifrostResponse, *schemas.BifrostError, error) {
	if resp == nil || resp.ChatResponse == nil {
		return resp, bifrostErr, nil
	}
	for _, choice := range resp.ChatResponse.Choices {
		content := choiceContent(choice)
		if content == nil || content.ContentStr == nil {
			continue
		}
		redacted, found := RedactSecrets(*content.ContentStr)
		if len(found) > 0 {
			content.ContentStr = &redacted
			log.Printf("guardrail secret_redaction redacted %s", strings.Join(found, ", "))
		}
	}
	return resp, bifrostErr, nil
}

func block(guardrail string, found []string) *schemas.LLMPluginShortCircuit {
	message := fmt.Sprintf("blocked by guardrail %s: %s", guardrail, strings.Join(found, ", "))
	log.Print(message)
	return &schemas.LLMPluginShortCircuit{Error: &schemas.BifrostError{
		Type:           schemas.Ptr("guardrail_" + guardrail),
		IsBifrostError: true,
		StatusCode:     schemas.Ptr(400),
		AllowFallbacks: schemas.Ptr(false),
		Error: &schemas.ErrorField{
			Type:    schemas.Ptr("guardrail_" + guardrail),
			Code:    schemas.Ptr(strings.Join(found, ",")),
			Message: message,
		},
	}}
}

func requestText(req *schemas.BifrostRequest) string {
	if req == nil || req.ChatRequest == nil {
		return ""
	}
	parts := []string{}
	for _, message := range req.ChatRequest.Input {
		if message.Content == nil {
			continue
		}
		if message.Content.ContentStr != nil {
			parts = append(parts, *message.Content.ContentStr)
		}
		for _, block := range message.Content.ContentBlocks {
			if block.Text != nil {
				parts = append(parts, *block.Text)
			}
		}
	}
	return strings.Join(parts, "\n")
}

func choiceContent(choice schemas.BifrostResponseChoice) *schemas.ChatMessageContent {
	if choice.ChatNonStreamResponseChoice == nil || choice.ChatNonStreamResponseChoice.Message == nil {
		return nil
	}
	return choice.ChatNonStreamResponseChoice.Message.Content
}
