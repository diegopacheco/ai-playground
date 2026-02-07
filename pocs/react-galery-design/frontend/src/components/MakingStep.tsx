import { useEffect } from "react";
import { useStore } from "@tanstack/react-store";
import { wizardStore, setProgress, setGeneratedHtml, goToStep } from "../store/wizard";
import { generate } from "../engine/generator";
import { ProgressBar } from "./ProgressBar";
import styles from "./MakingStep.module.css";

const statusMessages: [number, string][] = [
  [0, "Parsing your prompt..."],
  [10, "Selecting layout structure..."],
  [25, "Building components..."],
  [40, "Applying theme..."],
  [60, "Generating CSS..."],
  [75, "Assembling HTML..."],
  [90, "Finalizing your app..."],
];

function getStatusMessage(progress: number, styleName: string): string {
  let msg = statusMessages[0][1];
  for (const [threshold, text] of statusMessages) {
    if (progress >= threshold) {
      msg = text === "Applying theme..." ? `Applying ${styleName} theme...` : text;
    }
  }
  return msg;
}

export function MakingStep() {
  const progress = useStore(wizardStore, (s) => s.progress);
  const statusMessage = useStore(wizardStore, (s) => s.statusMessage);
  const prompt = useStore(wizardStore, (s) => s.prompt);
  const selectedStyle = useStore(wizardStore, (s) => s.selectedStyle);

  useEffect(() => {
    let current = 0;
    let done = false;
    const interval = setInterval(() => {
      if (done) return;
      current += 2 + Math.random() * 3;
      if (current >= 100) {
        current = 100;
        done = true;
        clearInterval(interval);

        const html = generate(prompt, selectedStyle);
        setGeneratedHtml(html);

        setTimeout(() => {
          goToStep(3);
        }, 500);
      }
      setProgress(Math.min(current, 100), getStatusMessage(current, selectedStyle));
    }, 80);

    return () => {
      clearInterval(interval);
    };
  }, [prompt, selectedStyle]);

  return (
    <div className={styles.container}>
      <div className={styles.title}>Building Your App</div>
      <ProgressBar progress={progress} statusMessage={statusMessage} />
    </div>
  );
}
