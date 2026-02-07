import { useStore } from "@tanstack/react-store";
import { wizardStore } from "./store/wizard";
import { TabBar } from "./components/TabBar";
import { PromptStep } from "./components/PromptStep";
import { MakingStep } from "./components/MakingStep";
import { PreviewStep } from "./components/PreviewStep";
import { SearchStep } from "./components/SearchStep";
import styles from "./App.module.css";

export default function App() {
  const currentStep = useStore(wizardStore, (s) => s.currentStep);

  return (
    <div className={styles.app}>
      <div className={styles.header}>
        <div className={styles.title}>Prompt Maker</div>
        <div className={styles.subtitle}>Describe your app, choose a style, and generate it</div>
      </div>
      <div className={styles.container}>
        <TabBar />
        {currentStep === 1 && <PromptStep />}
        {currentStep === 2 && <MakingStep />}
        {currentStep === 3 && <PreviewStep />}
        {currentStep === 4 && <SearchStep />}
      </div>
    </div>
  );
}
