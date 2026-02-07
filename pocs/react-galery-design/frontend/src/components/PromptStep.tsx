import { useForm } from "@tanstack/react-form";
import { wizardStore, startGeneration, setPrompt, setStyle } from "../store/wizard";
import type { UxStyle } from "../types";
import styles from "./PromptStep.module.css";

const uxStyles: { value: UxStyle; label: string }[] = [
  { value: "terminal", label: "Terminal" },
  { value: "modern", label: "Modern" },
  { value: "traditional", label: "Traditional" },
  { value: "brutalist", label: "Brutalist" },
  { value: "glassmorphism", label: "Glassmorphism" },
  { value: "neomorphism", label: "Neomorphism" },
  { value: "retro", label: "Retro" },
  { value: "minimalist", label: "Minimalist" },
  { value: "corporate", label: "Corporate" },
  { value: "playful", label: "Playful" },
  { value: "darkmode", label: "Dark Mode" },
  { value: "material", label: "Material" },
  { value: "flat", label: "Flat" },
];

export function PromptStep() {
  const form = useForm({
    defaultValues: {
      prompt: wizardStore.state.prompt || "",
      style: wizardStore.state.selectedStyle || ("modern" as UxStyle),
    },
    onSubmit: ({ value }) => {
      setPrompt(value.prompt);
      setStyle(value.style);
      startGeneration();
    },
  });

  return (
    <div className={styles.container}>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          e.stopPropagation();
          form.handleSubmit();
        }}
      >
        <form.Field name="prompt">
          {(field) => (
            <div>
              <label className={styles.label}>Describe the app you want to build</label>
              <textarea
                className={styles.textarea}
                placeholder="I want a landing page with a navigation bar, hero section with a headline, a features grid, pricing table, and a footer..."
                value={field.state.value}
                onChange={(e) => field.handleChange(e.target.value)}
              />
            </div>
          )}
        </form.Field>

        <form.Field name="style">
          {(field) => (
            <div className={styles.selectWrapper}>
              <label className={styles.label}>UX Style</label>
              <select
                className={styles.select}
                value={field.state.value}
                onChange={(e) => field.handleChange(e.target.value as UxStyle)}
              >
                {uxStyles.map((s) => (
                  <option key={s.value} value={s.value}>
                    {s.label}
                  </option>
                ))}
              </select>
            </div>
          )}
        </form.Field>

        <form.Subscribe selector={(s) => s.values.prompt}>
          {(prompt) => (
            <button
              type="submit"
              className={styles.generateBtn}
              disabled={!prompt.trim()}
            >
              Generate
            </button>
          )}
        </form.Subscribe>
      </form>
    </div>
  );
}
