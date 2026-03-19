import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Agent Werewolf",
  description: "Multi-agent social deduction game",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-gray-950 text-gray-100 min-h-screen">
        <nav className="border-b border-gray-800 px-6 py-4">
          <div className="max-w-6xl mx-auto flex items-center justify-between">
            <a href="/" className="text-xl font-bold text-red-500">Agent Werewolf</a>
            <div className="flex gap-4">
              <a href="/" className="text-gray-400 hover:text-white">New Game</a>
              <a href="/history" className="text-gray-400 hover:text-white">History</a>
            </div>
          </div>
        </nav>
        <main className="max-w-6xl mx-auto px-6 py-8">{children}</main>
      </body>
    </html>
  );
}
