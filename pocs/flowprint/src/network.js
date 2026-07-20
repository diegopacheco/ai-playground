(() => {
  if (window.__flowprintNetworkInstalled) return;
  window.__flowprintNetworkInstalled = true;

  const emit = payload => window.postMessage({ source: "flowprint-page", payload }, "*");
  const absoluteUrl = value => {
    try {
      return new URL(value, location.href).href;
    } catch {
      return String(value);
    }
  };

  const originalFetch = window.fetch;
  window.fetch = async (...args) => {
    const request = args[0];
    const options = args[1] || {};
    const method = String(options.method || request?.method || "GET").toUpperCase();
    const url = absoluteUrl(request?.url || request);
    const started = performance.now();
    try {
      const response = await originalFetch.apply(window, args);
      emit({ type: "network", transport: "fetch", method, url, status: response.status, duration: Math.round(performance.now() - started), failed: !response.ok, timestamp: Date.now() });
      return response;
    } catch (error) {
      emit({ type: "network", transport: "fetch", method, url, status: 0, duration: Math.round(performance.now() - started), failed: true, error: error.message, timestamp: Date.now() });
      throw error;
    }
  };

  const originalOpen = XMLHttpRequest.prototype.open;
  const originalSend = XMLHttpRequest.prototype.send;
  XMLHttpRequest.prototype.open = function(method, url, ...rest) {
    this.__flowprint = { method: String(method).toUpperCase(), url: absoluteUrl(url) };
    return originalOpen.call(this, method, url, ...rest);
  };
  XMLHttpRequest.prototype.send = function(...args) {
    const started = performance.now();
    this.addEventListener("loadend", () => {
      const request = this.__flowprint || { method: "GET", url: this.responseURL };
      emit({ type: "network", transport: "xhr", method: request.method, url: request.url, status: this.status, duration: Math.round(performance.now() - started), failed: this.status === 0 || this.status >= 400, timestamp: Date.now() });
    }, { once: true });
    return originalSend.apply(this, args);
  };

  const wrapHistory = method => {
    const original = history[method];
    history[method] = function(...args) {
      const result = original.apply(this, args);
      emit({ type: "navigation", url: location.href, timestamp: Date.now() });
      return result;
    };
  };

  wrapHistory("pushState");
  wrapHistory("replaceState");
  addEventListener("popstate", () => emit({ type: "navigation", url: location.href, timestamp: Date.now() }));
  addEventListener("hashchange", () => emit({ type: "navigation", url: location.href, timestamp: Date.now() }));
})();
