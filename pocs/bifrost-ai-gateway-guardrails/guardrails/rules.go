package main

import (
	"regexp"
	"sort"
)

type rule struct {
	name    string
	pattern *regexp.Regexp
	valid   func(string) bool
}

var piiRules = []rule{
	{name: "email", pattern: regexp.MustCompile(`[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}`)},
	{name: "us_ssn", pattern: regexp.MustCompile(`\b\d{3}-\d{2}-\d{4}\b`)},
	{name: "phone", pattern: regexp.MustCompile(`(?:\+?1[ .-]?)?\(?\b\d{3}\)?[ .-]\d{3}[ .-]\d{4}\b`)},
	{name: "credit_card", pattern: regexp.MustCompile(`\b\d(?:[ -]?\d){12,18}\b`), valid: luhn},
}

var injectionRules = []rule{
	{name: "ignore_instructions", pattern: regexp.MustCompile(`(?i)\b(ignore|disregard|forget)\s+(all\s+|any\s+)?(of\s+)?(the\s+|your\s+)?(previous|prior|above|earlier)\s+(instructions|prompts|rules|messages)`)},
	{name: "reveal_system_prompt", pattern: regexp.MustCompile(`(?i)\b(reveal|show|print|repeat|leak)\s+(me\s+)?(your|the)\s+(system|hidden|initial)\s+(prompt|instructions)`)},
	{name: "jailbreak_persona", pattern: regexp.MustCompile(`(?i)\b(you\s+are\s+now\s+(dan|in\s+developer\s+mode)|jailbreak|do\s+anything\s+now)\b`)},
}

var secretRules = []rule{
	{name: "aws_access_key", pattern: regexp.MustCompile(`\b(?:AKIA|ASIA)[0-9A-Z]{16}\b`)},
	{name: "openai_key", pattern: regexp.MustCompile(`\bsk-[A-Za-z0-9_-]{20,}`)},
	{name: "github_token", pattern: regexp.MustCompile(`\bgh[pousr]_[A-Za-z0-9]{36,}\b`)},
	{name: "private_key", pattern: regexp.MustCompile(`-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----`)},
}

func luhn(value string) bool {
	sum, digits, double := 0, 0, false
	for i := len(value) - 1; i >= 0; i-- {
		c := value[i]
		if c < '0' || c > '9' {
			continue
		}
		d := int(c - '0')
		if double {
			d *= 2
			if d > 9 {
				d -= 9
			}
		}
		sum += d
		digits++
		double = !double
	}
	return digits >= 13 && sum%10 == 0
}

func matches(rules []rule, text string) []string {
	found := []string{}
	for _, r := range rules {
		for _, hit := range r.pattern.FindAllString(text, -1) {
			if r.valid == nil || r.valid(hit) {
				found = append(found, r.name)
				break
			}
		}
	}
	sort.Strings(found)
	return found
}

func FindPII(text string) []string {
	return matches(piiRules, text)
}

func FindInjection(text string) []string {
	return matches(injectionRules, text)
}

func RedactSecrets(text string) (string, []string) {
	found := []string{}
	for _, r := range secretRules {
		if r.pattern.MatchString(text) {
			found = append(found, r.name)
			text = r.pattern.ReplaceAllString(text, "[REDACTED:"+r.name+"]")
		}
	}
	sort.Strings(found)
	return text, found
}
