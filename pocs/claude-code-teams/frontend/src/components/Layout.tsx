import { ReactNode } from "react";

interface LayoutProps {
  children: ReactNode;
}

export default function Layout({ children }: LayoutProps) {
  return (
    <div style={{
      maxWidth: "600px",
      margin: "0 auto",
      minHeight: "100vh",
      borderLeft: "1px solid #38444d",
      borderRight: "1px solid #38444d",
    }}>
      {children}
    </div>
  );
}
