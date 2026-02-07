import { useStore } from "@tanstack/react-store";
import { wizardStore, goToStep } from "../store/wizard";
import type { WizardStep } from "../types";
import styles from "./TabBar.module.css";

const tabs: { step: WizardStep; label: string }[] = [
  { step: 1, label: "1. Prompt" },
  { step: 2, label: "2. Making" },
  { step: 3, label: "3. Preview" },
  { step: 4, label: "4. Search" },
];

export function TabBar() {
  const currentStep = useStore(wizardStore, (s) => s.currentStep);

  function getTabClass(step: WizardStep): string {
    if (step === currentStep) return `${styles.tab} ${styles.active}`;
    if (step === 4) return `${styles.tab} ${styles.completed}`;
    if (step < currentStep) return `${styles.tab} ${styles.completed}`;
    return `${styles.tab} ${styles.disabled}`;
  }

  function handleClick(step: WizardStep) {
    if (step === 4 || step < currentStep) {
      goToStep(step);
    }
  }

  return (
    <div className={styles.tabBar}>
      {tabs.map((t) => (
        <button
          key={t.step}
          className={getTabClass(t.step)}
          onClick={() => handleClick(t.step)}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}
