import type { Metadata } from "next";
import type { ReactNode } from "react";

export const metadata: Metadata = {
  title: "qa2pw",
  description: "Plain English in. Real Playwright out.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body
        style={{
          margin: 0,
          fontFamily:
            '"Inter Tight", -apple-system, BlinkMacSystemFont, sans-serif',
          background: "#FAFAFA",
          color: "#18181B",
        }}
      >
        {children}
      </body>
    </html>
  );
}
