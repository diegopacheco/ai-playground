import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { CalculatorApp } from "./CalculatorApp";
import "./view.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <CalculatorApp />
  </StrictMode>,
);
