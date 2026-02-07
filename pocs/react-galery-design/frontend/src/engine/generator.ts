import type { UxStyle, StyleConfig, FragmentId } from "../types";
import { parsePrompt } from "./parser";
import { generateBaseCSS } from "./css/base";
import { generateResponsiveCSS } from "./css/responsive";
import { terminalStyle } from "./styles/terminal";
import { modernStyle } from "./styles/modern";
import { traditionalStyle } from "./styles/traditional";
import { brutalistStyle } from "./styles/brutalist";
import { glassmorphismStyle } from "./styles/glassmorphism";
import { neomorphismStyle } from "./styles/neomorphism";
import { retroStyle } from "./styles/retro";
import { minimalistStyle } from "./styles/minimalist";
import { corporateStyle } from "./styles/corporate";
import { playfulStyle } from "./styles/playful";
import { darkmodeStyle } from "./styles/darkmode";
import { materialStyle } from "./styles/material";
import { flatStyle } from "./styles/flat";
import { generateNavbar } from "./fragments/navbar";
import { generateHero } from "./fragments/hero";
import { generateCardGrid } from "./fragments/cardGrid";
import { generateFooter } from "./fragments/footer";
import { generateFeatures } from "./fragments/features";
import { generateContactForm } from "./fragments/contactForm";
import { generateSidebar } from "./fragments/sidebar";
import { generateTable } from "./fragments/table";
import { generateGallery } from "./fragments/gallery";
import { generateStats } from "./fragments/stats";
import { generatePricing } from "./fragments/pricing";
import { generateCta } from "./fragments/cta";

const styleMap: Record<UxStyle, StyleConfig> = {
  terminal: terminalStyle,
  modern: modernStyle,
  traditional: traditionalStyle,
  brutalist: brutalistStyle,
  glassmorphism: glassmorphismStyle,
  neomorphism: neomorphismStyle,
  retro: retroStyle,
  minimalist: minimalistStyle,
  corporate: corporateStyle,
  playful: playfulStyle,
  darkmode: darkmodeStyle,
  material: materialStyle,
  flat: flatStyle,
};

const fragmentMap: Record<FragmentId, () => string> = {
  navbar: generateNavbar,
  hero: generateHero,
  cardGrid: generateCardGrid,
  footer: generateFooter,
  features: generateFeatures,
  contactForm: generateContactForm,
  sidebar: generateSidebar,
  table: generateTable,
  gallery: generateGallery,
  stats: generateStats,
  pricing: generatePricing,
  cta: generateCta,
};

function generateCSSVariables(config: StyleConfig): string {
  return (
    ":root {\n" +
    Object.entries(config.cssVariables)
      .map(([key, val]) => `  ${key}: ${val};`)
      .join("\n") +
    "\n}\n"
  );
}

export function generate(prompt: string, style: UxStyle): string {
  const config = styleMap[style];
  const fragmentIds = parsePrompt(prompt);

  const cssVars = generateCSSVariables(config);
  const baseCSS = generateBaseCSS();
  const responsiveCSS = generateResponsiveCSS();

  const htmlFragments = fragmentIds
    .map((id) => fragmentMap[id]())
    .join("\n\n");

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Generated App</title>
  <style>
${cssVars}
${baseCSS}
${config.bodyStyles}
${config.componentOverrides}
${responsiveCSS}
  </style>
</head>
<body>
${htmlFragments}
</body>
</html>`;
}
