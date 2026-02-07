export function generateBaseCSS(): string {
  return `
*, *::before, *::after {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

html {
  font-size: 16px;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

body {
  font-family: var(--font);
  color: var(--text);
  background-color: var(--bg);
  line-height: 1.6;
}

a {
  color: var(--primary);
  text-decoration: none;
}

a:hover {
  opacity: 0.8;
}

button {
  cursor: pointer;
  border: none;
  font-family: inherit;
  font-size: 1rem;
}

img {
  max-width: 100%;
  height: auto;
}

input, textarea, select {
  font-family: inherit;
  font-size: 1rem;
}

h1, h2, h3, h4, h5, h6 {
  line-height: 1.2;
}

section {
  padding: var(--spacing) 0;
}

.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 var(--spacing);
}
`;
}
