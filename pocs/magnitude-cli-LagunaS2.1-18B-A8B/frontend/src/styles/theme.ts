export const theme = {
  colors: {
    primary: "#2563eb",
    primaryHover: "#1d4ed8",
    secondary: "#64748b",
    background: "#f8fafc",
    surface: "#ffffff",
    border: "#e2e8f0",
    text: "#1e293b",
    textSecondary: "#64748b",
    success: "#10b981",
    danger: "#ef4444",
    warning: "#f59e0b",
    shadow: "0 1px 3px rgba(0, 0, 0, 0.1)",
    shadowHover: "0 4px 6px rgba(0, 0, 0, 0.15)",
  },
  spacing: {
    xs: "0.25rem",
    sm: "0.5rem",
    md: "1rem",
    lg: "1.5rem",
    xl: "2rem",
    xxl: "3rem",
  },
  borderRadius: {
    sm: "0.25rem",
    md: "0.5rem",
    lg: "0.75rem",
    xl: "1rem",
  },
  fontSize: {
    xs: "0.75rem",
    sm: "0.875rem",
    md: "1rem",
    lg: "1.125rem",
    xl: "1.25rem",
    xxl: "1.5rem",
    title: "2rem",
  },
  breakpoints: {
    mobile: "480px",
    tablet: "768px",
    desktop: "1024px",
  },
} as const;

export type AppTheme = typeof theme;
