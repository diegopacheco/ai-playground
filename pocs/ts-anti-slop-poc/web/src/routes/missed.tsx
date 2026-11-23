import { createFileRoute } from "@tanstack/react-router";
import { ReportView } from "../features/report/components/ReportView.tsx";
import type { Report } from "../features/report/types.ts";

const missed = (report: Report) => report.samples.filter((s) => s.verdict === "missed");

export const Route = createFileRoute("/missed")({
  component: () => <ReportView select={missed} withSummary={false} />,
});
