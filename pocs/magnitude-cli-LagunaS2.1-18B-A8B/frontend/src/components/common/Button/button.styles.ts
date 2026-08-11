import styled from "styled-components";

export type ButtonVariant = "primary" | "secondary" | "danger" | "ghost";
export type ButtonSize = "sm" | "md" | "lg";

interface ButtonProps {
  variant?: ButtonVariant;
  size?: ButtonSize;
  fullWidth?: boolean;
}

const variantColors = {
  primary: {
    bg: "var(--color-primary, #2563eb)",
    hover: "var(--color-primary-hover, #1d4ed8)",
    color: "#ffffff",
  },
  secondary: {
    bg: "var(--color-secondary, #64748b)",
    hover: "var(--color-secondary-hover, #475569)",
    color: "#ffffff",
  },
  danger: {
    bg: "var(--color-danger, #ef4444)",
    hover: "var(--color-danger-hover, #dc2626)",
    color: "#ffffff",
  },
  ghost: {
    bg: "transparent",
    hover: "rgba(37, 99, 235, 0.08)",
    color: "var(--color-primary, #2563eb)",
  },
};

const sizeStyles = {
  sm: { padding: "0.375rem 0.75rem", fontSize: "0.875rem" },
  md: { padding: "0.5rem 1rem", fontSize: "1rem" },
  lg: { padding: "0.625rem 1.25rem", fontSize: "1.125rem" },
};

export const StyledButton = styled.button<ButtonProps>`
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  border-radius: ${({ theme }) => theme.borderRadius.md};
  font-weight: 500;
  transition: all 0.2s ease;
  cursor: pointer;
  width: ${({ fullWidth }) => (fullWidth ? "100%" : "auto")};

  ${({ variant = "primary", theme }) => {
    const colors = variantColors[variant];
    return `
      background-color: ${colors.bg};
      color: ${colors.color};
      border: 1px solid ${colors.bg};

      &:hover {
        background-color: ${colors.hover};
        border-color: ${colors.hover};
        transform: translateY(-1px);
      }

      &:active {
        transform: translateY(0);
      }

      &:disabled {
        opacity: 0.5;
        cursor: not-allowed;
        transform: none;
      }
    `;
  }}

  ${({ size = "md" }) => {
    const s = sizeStyles[size];
    return `
      padding: ${s.padding};
      font-size: ${s.fontSize};
    `;
  }}
`;
