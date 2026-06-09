package main

import (
	"context"
	"fmt"
	"strings"

	"github.com/openai/openai-go/v2"
	"speechtomap/osm"
)

const maxIterations = 6

const model = "gpt-5.4-mini"

const systemPrompt = `You are a geospatial assistant. The user's current location and local time are given to you.
Use the tools to find places and routes that answer the request.
Prefer results close to the user. When the user says "open now", check each place's opening_hours against the given local time and exclude places that are clearly closed.
When the user says "walkable" or asks to walk, use the foot routing profile and prefer places within a short walking distance.
When it helps the user reach a chosen place, call get_route from the user's location to that place.
The channel is one-shot voice, so make reasonable defaults (about a 2 km radius, walking) rather than asking the user back.
Always answer concisely in plain language, naming the closest place and its distance or walking time when known.`

type agentResult struct {
	Answer string
	Places []osm.Place
	Route  *osm.Route
}

func runAgent(ctx context.Context, client openai.Client, req queryRequest) (agentResult, error) {
	userMsg := fmt.Sprintf("User request: %s\nUser location: lat=%f, lon=%f\nLocal time: %s",
		req.Text, req.Lat, req.Lon, req.Now)

	messages := []openai.ChatCompletionMessageParamUnion{
		openai.SystemMessage(systemPrompt),
		openai.UserMessage(userMsg),
	}

	var result agentResult

	for range maxIterations {
		completion, err := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
			Model:    model,
			Messages: messages,
			Tools:    toolSchemas(),
		})
		if err != nil {
			return result, err
		}
		if len(completion.Choices) == 0 {
			return result, fmt.Errorf("model returned no choices")
		}
		msg := completion.Choices[0].Message
		messages = append(messages, msg.ToParam())

		if len(msg.ToolCalls) == 0 {
			result.Answer = strings.TrimSpace(msg.Content)
			return result, nil
		}

		for _, tc := range msg.ToolCalls {
			res := dispatchTool(ctx, tc.Function.Name, tc.Function.Arguments)
			if len(res.places) > 0 {
				result.Places = res.places
			}
			if res.route != nil {
				result.Route = res.route
			}
			messages = append(messages, openai.ToolMessage(res.content, tc.ID))
		}
	}

	final, err := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Model:    model,
		Messages: messages,
	})
	if err == nil && len(final.Choices) > 0 {
		result.Answer = strings.TrimSpace(final.Choices[0].Message.Content)
	} else {
		result.Answer = "I could not complete the request within the tool-call limit."
	}
	return result, nil
}
