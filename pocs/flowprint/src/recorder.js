(() => {
  let recording = false;
  let lastClickAt = 0;

  const cssEscape = value => {
    if (globalThis.CSS?.escape) return CSS.escape(value);
    return String(value).replace(/[^a-zA-Z0-9_-]/g, character => `\\${character}`);
  };

  const cssString = value => String(value).replace(/\\/g, "\\\\").replace(/"/g, "\\\"");

  const roleFor = element => {
    const explicit = element.getAttribute("role");
    if (explicit) return explicit;
    if (element instanceof HTMLTextAreaElement) return "textbox";
    if (element instanceof HTMLButtonElement) return "button";
    if (element instanceof HTMLAnchorElement && element.hasAttribute("href")) return "link";
    if (element instanceof HTMLSelectElement) return "combobox";
    if (element instanceof HTMLInputElement) {
      if (["button", "submit", "reset", "image"].includes(element.type)) return "button";
      if (["checkbox", "radio", "range", "search"].includes(element.type)) return element.type === "search" ? "searchbox" : element.type;
      return "textbox";
    }
    return "";
  };

  const locatorFor = element => {
    const testId = element.getAttribute("data-testid");
    if (testId) return { kind: "testId", value: testId };
    const label = element.labels?.[0]?.textContent?.trim().replace(/\s+/g, " ");
    if (label) return { kind: "label", value: label };
    const ariaLabel = element.getAttribute("aria-label");
    const accessibleRole = roleFor(element);
    if (ariaLabel && accessibleRole) return { kind: "role", role: accessibleRole, value: ariaLabel };
    if (ariaLabel) return { kind: "label", value: ariaLabel };
    if (element.id) return { kind: "css", value: `#${cssEscape(element.id)}` };
    const name = element.getAttribute("name");
    if (name) return { kind: "css", value: `${element.tagName.toLowerCase()}[name="${cssString(name)}"]` };
    const role = accessibleRole;
    const text = element.textContent?.trim().replace(/\s+/g, " ").slice(0, 80);
    if (role && text) return { kind: "role", role, value: text };
    const parts = [];
    let current = element;
    while (current && current.nodeType === Node.ELEMENT_NODE && parts.length < 5) {
      let part = current.tagName.toLowerCase();
      const classes = [...current.classList].filter(value => /^[a-zA-Z][\w-]*$/.test(value)).slice(0, 2);
      if (classes.length) part += `.${classes.map(cssEscape).join(".")}`;
      const siblings = current.parentElement ? [...current.parentElement.children].filter(child => child.tagName === current.tagName) : [];
      if (siblings.length > 1) part += `:nth-of-type(${siblings.indexOf(current) + 1})`;
      parts.unshift(part);
      current = current.parentElement;
    }
    return { kind: "css", value: parts.join(" > ") };
  };

  const send = step => {
    if (!recording) return;
    chrome.runtime.sendMessage({ type: "flowprint:record", step }).catch(() => {});
  };

  addEventListener("click", event => {
    const element = event.target instanceof Element ? event.target.closest("button, a, input, select, textarea, [role], [data-testid]") || event.target : null;
    if (!element) return;
    const inputType = element instanceof HTMLInputElement ? element.type : "";
    if (["text", "email", "password", "search", "tel", "url", "number", "date", "time"].includes(inputType) || element instanceof HTMLTextAreaElement || element instanceof HTMLSelectElement) return;
    lastClickAt = Date.now();
    send({ type: "click", locator: locatorFor(element), text: element.textContent?.trim().slice(0, 120) || "", timestamp: Date.now() });
  }, true);

  addEventListener("change", event => {
    const element = event.target;
    if (!(element instanceof HTMLInputElement || element instanceof HTMLTextAreaElement || element instanceof HTMLSelectElement)) return;
    if (element instanceof HTMLInputElement && ["checkbox", "radio"].includes(element.type)) return;
    const value = element instanceof HTMLInputElement && element.type === "password" ? "[redacted]" : element.value;
    send({ type: element instanceof HTMLSelectElement ? "select" : "fill", locator: locatorFor(element), value, timestamp: Date.now() });
  }, true);

  addEventListener("keydown", event => {
    if (event.key !== "Enter") return;
    const element = event.target;
    if (!(element instanceof HTMLInputElement) || !element.form) return;
    lastClickAt = Date.now();
    send({ type: "press", locator: locatorFor(element), value: "Enter", timestamp: Date.now() });
  }, true);

  addEventListener("submit", event => {
    const form = event.target;
    if (form instanceof HTMLFormElement && Date.now() - lastClickAt > 500) send({ type: "submit", locator: locatorFor(form), timestamp: Date.now() });
  }, true);

  addEventListener("message", event => {
    if (event.source !== window || event.data?.source !== "flowprint-page") return;
    send(event.data.payload);
  });

  chrome.runtime.onMessage.addListener(message => {
    if (message.type === "flowprint:set-recording") recording = message.recording;
  });

  chrome.runtime.sendMessage({ type: "flowprint:ready" }).then(response => {
    recording = Boolean(response?.recording);
  }).catch(() => {});
})();
