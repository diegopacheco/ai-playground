import { createFileRoute } from "@tanstack/react-router";
import { JudgePanel } from "../features/judge/components/JudgePanel";

export const Route = createFileRoute("/judge")({ component: JudgePanel });
