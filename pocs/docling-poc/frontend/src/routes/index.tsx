import { createFileRoute } from "@tanstack/react-router";
import { UploadPanel } from "../features/conversions/components/UploadPanel";

export const Route = createFileRoute("/")({
  component: ConvertPage,
});

function ConvertPage() {
  return <UploadPanel />;
}
