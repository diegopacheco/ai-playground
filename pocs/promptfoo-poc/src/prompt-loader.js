const fs = require('fs');
const path = require('path');

const PROMPTS_DIR = path.join(__dirname, '..', 'prompts');
const PARAM = /\$([a-z][a-z0-9_]*)/g;

function parse(name) {
  const file = path.join(PROMPTS_DIR, `${name}.md`);
  const raw = fs.readFileSync(file, 'utf8');
  const parts = raw.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
  if (!parts) throw new Error(`prompt ${name}: missing --- header ---`);

  const header = {};
  for (const line of parts[1].split('\n')) {
    const sep = line.indexOf(':');
    if (sep > 0) header[line.slice(0, sep).trim()] = line.slice(sep + 1).trim();
  }
  if (header.name !== name) throw new Error(`prompt ${name}: header name is "${header.name}"`);

  const declared = (header.params || '')
    .split(',')
    .map((p) => p.trim().replace(/^\$/, ''))
    .filter(Boolean);

  return { declared, template: parts[2].trim() };
}

function render(name, vars) {
  const { declared, template } = parse(name);

  for (const [, param] of template.matchAll(PARAM)) {
    if (!declared.includes(param)) throw new Error(`prompt ${name}: $${param} is used but not declared`);
  }
  for (const param of declared) {
    if (vars[param] === undefined || vars[param] === null || vars[param] === '') {
      throw new Error(`prompt ${name}: no value for $${param}`);
    }
  }

  return template.replace(PARAM, (_, param) => String(vars[param]));
}

module.exports = { render };
