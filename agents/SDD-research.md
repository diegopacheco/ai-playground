# SDD (Spec-Driven Development) - Short Description

SDD is writing detailed specifications in natural language (typically markdown) before code creation, where
specs serve as the authoritative reference guiding AI agents. It comes in three levels:

1. Spec-First: Specs created then discarded
2. Spec-Anchored: Specs persist throughout feature evolution
3. Spec-as-Source: Specs are primary artifacts; humans never edit code directly

## SDD vs Few-Shot/Short Prompts - Comparison

| Aspect             | SDD                           | Few-Shot/Short Prompts         |
|--------------------|-------------------------------|--------------------------------|
| Context Usage      | High (often 1000s of tokens)  | Low (typically 100-500 tokens) |
| Setup Cost         | High initial investment       | Minimal setup                  |
| Task Specification | Comprehensive, detailed       | Minimal, example-based         |
| Maintenance        | Specs require ongoing updates | No maintenance needed          |
| Flexibility        | Can become rigid              | Highly adaptable               |

## Is SDD Always Better? NO

Research shows diminishing returns after 2-3 examples for many tasks. The "over-prompting phenomenon" reveals
that excessive examples paradoxically degrade performance in certain LLMs.

Trade-offs:

SDD Advantages:
- Better for complex, multi-step features
- Reduces ambiguity for large codebases
- Can overcome pre-training biases with sufficient context
- Acts as "single source of truth"

SDD Disadvantages:
- Context window bloat - teams documenting everything find agents "choking on sheer volume"
- Review burden - reading extensive markdown is more tedious than reviewing code
- Instruction adherence issues - AI agents frequently ignore or over-interpret specs
- Historical parallels to failed Model-Driven Development approaches
- Workflow doesn't scale - one-size-fits-all unsuitable for varying problem sizes

Few-Shot/Short Prompts Advantages:
- Token-efficient
- Fast iteration
- Works well for simple, focused tasks
- No maintenance overhead
- Better for exploratory work

Few-Shot/Short Prompts Disadvantages:
- Limited context for complex tasks
- Requires clearer mental model from developer
- Less reproducible across different sessions

## Key Research Findings

1. Many-shot ICL can match fine-tuning performance with sufficient examples
2. Context window limitations remain critical - RAG/intelligent spec selection needed
3. Over-prompting is real - more isn't always better
4. Long context models now outperform RAG in many scenarios (GPT-4O, Gemini-1.5-Pro)

Your Concern About Documentation Gaps: CONFIRMED

Your intuition is correct. Research found:

- "Integration wall" was a silent killer of AI agent projects until late 2024
- MCP (Model Context Protocol) introduced by Anthropic specifically to address the "missing skill layer"
- Prior to MCP: hours spent studying docs, writing boilerplate, handling auth per tool
- Composio MCP emerged providing out-of-the-box access to 100+ tools with unified APIs
- Microsoft created "MCP for Beginners" curriculum in early 2025 to fill documentation gaps

## Recommendation

Use SDD when:
- Building large, complex features
- Need reproducibility across team
- Feature requires extensive domain knowledge

Use Few-Shot/Short Prompts when:
- Quick iterations needed
- Simple, focused tasks
- Exploratory work
- Context window budget is limited

Hybrid approach works best: Brief specs (300-500 tokens) + 2-3 examples for optimal token efficiency and performance.

The Martin Fowler article's skepticism is warranted - SDD tools remain immature with unresolved workflow scalability issues.