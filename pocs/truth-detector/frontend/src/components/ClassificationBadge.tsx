interface ClassificationBadgeProps {
  classification: "DEEP" | "DECENT" | "SHALLOW";
}

const styles: Record<string, string> = {
  DEEP: "bg-green-500/20 text-green-400 border border-green-500/30",
  DECENT: "bg-yellow-500/20 text-yellow-400 border border-yellow-500/30",
  SHALLOW: "bg-red-500/20 text-red-400 border border-red-500/30",
};

function ClassificationBadge({ classification }: ClassificationBadgeProps) {
  return (
    <span
      className={`inline-block px-2 py-0.5 rounded-full text-xs font-semibold uppercase ${styles[classification]}`}
    >
      {classification}
    </span>
  );
}

export default ClassificationBadge;
