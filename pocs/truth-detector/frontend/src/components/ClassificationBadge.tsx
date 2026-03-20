interface ClassificationBadgeProps {
  classification: "DEEP" | "DECENT" | "SHALLOW";
}

const styles: Record<string, string> = {
  DEEP: "bg-green-100 text-green-700 border border-green-300",
  DECENT: "bg-yellow-100 text-yellow-700 border border-yellow-300",
  SHALLOW: "bg-red-100 text-red-700 border border-red-300",
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
