import { createFileRoute } from "@tanstack/react-router";
import { ReportView } from "../features/report/components/ReportView.tsx";
import type { Report } from "../features/report/types.ts";

const all = (report: Report) => report.samples;

export const Route = createFileRoute("/")({
  component: () => <ReportView select={all} withSummary />,
});
