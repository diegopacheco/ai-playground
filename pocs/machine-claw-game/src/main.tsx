import { createRoot } from "react-dom/client";
import { App } from "./ui/App";
import "./ui/styles.css";

const root = document.getElementById("root");
if (!root) throw new Error("root element is missing");
createRoot(root).render(<App />);
