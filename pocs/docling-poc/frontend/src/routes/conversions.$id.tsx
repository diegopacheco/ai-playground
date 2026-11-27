import { createFileRoute } from "@tanstack/react-router";
import { DocumentTabs } from "../features/conversions/components/DocumentTabs";
import { StatsBar } from "../features/conversions/components/StatsBar";
import { StatusBadge } from "../features/conversions/components/StatusBadge";
import { formatDuration } from "../features/conversions/format";
import { useConversion } from "../features/conversions/hooks";

export const Route = createFileRoute("/conversions/$id")({
  component: ConversionPage,
});

function ConversionPage() {
  const { id } = Route.useParams();
  const { data, isPending, error } = useConversion(id);
  if (isPending) return <p className="muted">Loading...</p>;
  if (error) return <p className="error">{error.message}</p>;
  return (
    <section className="stack">
      <div className="card head">
        <div>
          <h2>{data.filename}</h2>
          <p className="muted">
            {data.format} - converted in {formatDuration(data.durationMs)} - OCR {data.ocr ? "on" : "off"}
          </p>
        </div>
        <StatusBadge status={data.status} />
      </div>
      {data.error && <p className="error">{data.error}</p>}
      {data.document && (
        <>
          <StatsBar stats={data.document.stats} />
          <DocumentTabs document={data.document} />
        </>
      )}
    </section>
  );
}
