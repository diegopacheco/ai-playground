import type { ButtonHTMLAttributes, ReactNode } from "react";
import "./Button.css";

export type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  children: ReactNode;
}

export function Button({ variant = "secondary", children, className, ...rest }: ButtonProps) {
  return (
    <button className={["ds-button", `ds-button-${variant}`, className].filter(Boolean).join(" ")} {...rest}>
      {children}
    </button>
  );
}
