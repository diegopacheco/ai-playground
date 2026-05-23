import type { Selector } from "./types.ts";

export interface BrowserAdapter {
  goto(url: string): Promise<void>;
  screenshot(): Promise<string>;
  click(selector: Selector): Promise<void>;
  type(selector: Selector, text: string): Promise<void>;
  waitFor(selector: Selector): Promise<void>;
  assertText(selector: Selector, text: string): Promise<void>;
  currentUrl(): Promise<string>;
}

export function selectorLabel(selector: Selector): string {
  switch (selector.kind) {
    case "role":
      return selector.name
        ? `role=${selector.role} name=${JSON.stringify(selector.name)}`
        : `role=${selector.role}`;
    case "placeholder":
      return `placeholder=${JSON.stringify(selector.text)}`;
    case "text":
      return `text=${JSON.stringify(selector.text)}`;
    case "label":
      return `label=${JSON.stringify(selector.text)}`;
    case "test_id":
      return `test_id=${JSON.stringify(selector.id)}`;
  }
}
