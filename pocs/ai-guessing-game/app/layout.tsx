import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Mind Reader — AI Guessing Game",
  description: "Think of something. The AI gets 5 guesses. HOT or COLD is all you give.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <header className="topbar">
          <Link href="/" className="brand">
            <span className="brand-emoji">🔮</span>
            <span>Mind Reader</span>
          </Link>
          <nav className="nav">
            <Link href="/">Play</Link>
            <Link href="/history">History</Link>
          </nav>
        </header>
        <main className="main">{children}</main>
      </body>
    </html>
  );
}
