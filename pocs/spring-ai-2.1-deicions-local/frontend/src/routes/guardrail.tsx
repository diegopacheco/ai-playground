import { createFileRoute } from "@tanstack/react-router";
import { GuardrailPanel } from "../features/guardrail/components/GuardrailPanel";

export const Route = createFileRoute("/guardrail")({ component: GuardrailPanel });
