import { Store } from "@tanstack/store";
import type { WizardState, WizardStep, UxStyle } from "../types";

const initialState: WizardState = {
  currentStep: 1,
  prompt: "",
  selectedStyle: "modern",
  progress: 0,
  statusMessage: "",
  generatedHtml: "",
  isGenerating: false,
};

export const wizardStore = new Store<WizardState>(initialState);

export function goToStep(step: WizardStep) {
  wizardStore.setState((prev) => ({ ...prev, currentStep: step }));
}

export function setPrompt(text: string) {
  wizardStore.setState((prev) => ({ ...prev, prompt: text }));
}

export function setStyle(style: UxStyle) {
  wizardStore.setState((prev) => ({ ...prev, selectedStyle: style }));
}

export function setProgress(progress: number, statusMessage: string) {
  wizardStore.setState((prev) => ({ ...prev, progress, statusMessage }));
}

export function setGeneratedHtml(html: string) {
  wizardStore.setState((prev) => ({
    ...prev,
    generatedHtml: html,
    isGenerating: false,
    progress: 100,
    statusMessage: "Done!",
  }));
}

export function startGeneration() {
  wizardStore.setState((prev) => ({
    ...prev,
    isGenerating: true,
    currentStep: 2,
    progress: 0,
    statusMessage: "Parsing your prompt...",
  }));
}

export function reset() {
  wizardStore.setState(() => ({ ...initialState }));
}
