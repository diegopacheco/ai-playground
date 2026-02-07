export type UxStyle =
  | "terminal"
  | "modern"
  | "traditional"
  | "brutalist"
  | "glassmorphism"
  | "neomorphism"
  | "retro"
  | "minimalist"
  | "corporate"
  | "playful"
  | "darkmode"
  | "material"
  | "flat";

export type WizardStep = 1 | 2 | 3 | 4;

export type SavedSolution = {
  id: string;
  prompt: string;
  style: UxStyle;
  html: string;
  createdAt: number;
};

export type WizardState = {
  currentStep: WizardStep;
  prompt: string;
  selectedStyle: UxStyle;
  progress: number;
  statusMessage: string;
  generatedHtml: string;
  isGenerating: boolean;
};

export type StyleConfig = {
  name: string;
  cssVariables: Record<string, string>;
  bodyStyles: string;
  componentOverrides: string;
};

export type FragmentId =
  | "navbar"
  | "hero"
  | "cardGrid"
  | "footer"
  | "features"
  | "contactForm"
  | "sidebar"
  | "table"
  | "gallery"
  | "stats"
  | "pricing"
  | "cta";
