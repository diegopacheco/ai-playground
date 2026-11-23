import { useReport } from "../hooks.ts";
import type { Report, SampleResult } from "../types.ts";
import { RuleCoverageList } from "./RuleCoverageList.tsx";
import { SampleTable } from "./SampleTable.tsx";
import { Summary } from "./Summary.tsx";

type ReportViewProps = {
  readonly select: (report: Report) => readonly SampleResult[];
  readonly withSummary: boolean;
};

export function ReportView({ select, withSummary }: ReportViewProps) {
  const { data, isPending, error } = useReport();

  if (isPending) return <p className="muted">Running oxlint with anti-slop on slop/ ...</p>;
  if (error) throw error;

  return (
    <>
      {withSummary && <Summary report={data} />}
      {withSummary && <RuleCoverageList rules={data.rules} />}
      <SampleTable samples={select(data)} />
    </>
  );
}
