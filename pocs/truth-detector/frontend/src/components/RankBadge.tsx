interface RankBadgeProps {
  rank: number;
}

function RankBadge({ rank }: RankBadgeProps) {
  let style = "text-gray-500";
  if (rank === 1) style = "text-yellow-400 font-bold";
  if (rank === 2) style = "text-gray-400 font-bold";
  if (rank === 3) style = "text-amber-600 font-bold";

  return <span className={`text-sm ${style}`}>#{rank}</span>;
}

export default RankBadge;
