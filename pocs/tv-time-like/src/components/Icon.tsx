type IconName = "search" | "library" | "metrics" | "check" | "plus" | "play" | "clock" | "star" | "arrow" | "upload" | "close" | "film" | "tv" | "trash"

const paths: Record<IconName, React.ReactNode> = {
  search: <><circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/></>,
  library: <><path d="M4 5h16v15H4z"/><path d="M8 5V3h8v2M8 10h8M8 14h5"/></>,
  metrics: <><path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/></>,
  check: <path d="m5 12 4 4L19 6"/>,
  plus: <path d="M12 5v14M5 12h14"/>,
  play: <path d="m8 5 11 7-11 7z"/>,
  clock: <><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></>,
  star: <path d="m12 3 2.8 5.7 6.2.9-4.5 4.4 1.1 6.2-5.6-3-5.6 3 1.1-6.2L3 9.6l6.2-.9z"/>,
  arrow: <path d="M5 12h14M14 7l5 5-5 5"/>,
  upload: <><path d="M12 16V4M7 9l5-5 5 5"/><path d="M5 15v5h14v-5"/></>,
  close: <path d="m6 6 12 12M18 6 6 18"/>,
  film: <><rect x="3" y="5" width="18" height="14" rx="2"/><path d="M7 5v14M17 5v14M3 9h4M17 9h4M3 15h4M17 15h4"/></>,
  tv: <><rect x="3" y="6" width="18" height="14" rx="2"/><path d="m8 2 4 4 4-4"/></>,
  trash: <><path d="M5 7h14M9 7V4h6v3M7 7l1 13h8l1-13"/><path d="M10 11v5M14 11v5"/></>
}

export function Icon({ name, size = 20 }: { name: IconName; size?: number }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>
}
