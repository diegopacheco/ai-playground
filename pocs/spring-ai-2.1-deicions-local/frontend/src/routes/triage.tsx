import { createFileRoute } from "@tanstack/react-router";
import { TriagePanel } from "../features/triage/components/TriagePanel";

export const Route = createFileRoute("/triage")({ component: TriagePanel });
