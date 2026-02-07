import { Store } from "@tanstack/store";
import type { WizardState, WizardStep, UxStyle, SavedSolution } from "../types";

const STORAGE_KEY = "prompt-maker-solutions";

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
  const state = wizardStore.state;
  const solution: SavedSolution = {
    id: crypto.randomUUID(),
    prompt: state.prompt,
    style: state.selectedStyle,
    html,
    createdAt: Date.now(),
  };
  saveSolution(solution);
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

export function loadSolutionForPreview(html: string) {
  wizardStore.setState((prev) => ({
    ...prev,
    generatedHtml: html,
    currentStep: 3,
  }));
}

function saveSolution(solution: SavedSolution) {
  const existing = getSolutions();
  existing.unshift(solution);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(existing));
}

export function getSolutions(): SavedSolution[] {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) return [];
  return JSON.parse(raw);
}

export function deleteSolution(id: string) {
  const existing = getSolutions().filter((s) => s.id !== id);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(existing));
}
