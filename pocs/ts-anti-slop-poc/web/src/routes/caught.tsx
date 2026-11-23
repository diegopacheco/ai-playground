import { createFileRoute } from "@tanstack/react-router";
import { ReportView } from "../features/report/components/ReportView.tsx";
import type { Report } from "../features/report/types.ts";

const caught = (report: Report) => report.samples.filter((s) => s.verdict === "caught");

export const Route = createFileRoute("/caught")({
  component: () => <ReportView select={caught} withSummary={false} />,
});
