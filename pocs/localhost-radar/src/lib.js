(function(root) {
  const shellQuote = value => {
    const text = String(value);
    return /^[a-zA-Z0-9_.-]+$/.test(text) ? text : `'${text.replace(/'/g, `'\\''`)}'`;
  };

  const containerCommand = container => `podman exec -it ${shellQuote(container.name || container.id)} sh`;
  const serviceUrl = service => `http://localhost:${Number(service.port)}`;
  const curlCommand = url => `curl -i ${shellQuote(url)}`;
  const duration = milliseconds => `${Math.max(0, Math.round(Number(milliseconds) || 0))} ms`;
  const statusTone = status => Number(status) >= 400 || Number(status) === 0 ? "danger" : Number(status) >= 300 ? "warn" : "good";

  const api = { containerCommand, curlCommand, duration, serviceUrl, shellQuote, statusTone };
  root.LocalhostRadarLib = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
