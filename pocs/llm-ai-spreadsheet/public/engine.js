function sentinelError(s) {
  const e = new Error(s);
  e.sentinel = s;
  return e;
}

function colToNum(col) {
  let n = 0;
  for (const ch of col) n = n * 26 + (ch.charCodeAt(0) - 64);
  return n;
}

function numToCol(n) {
  let s = '';
  while (n > 0) {
    const r = (n - 1) % 26;
    s = String.fromCharCode(65 + r) + s;
    n = Math.floor((n - 1) / 26);
  }
  return s;
}

function splitRef(ref) {
  const m = ref.match(/^([A-Z]+)([0-9]+)$/);
  return { col: colToNum(m[1]), row: Number(m[2]) };
}

function expandRange(from, to) {
  const a = splitRef(from);
  const b = splitRef(to);
  const c1 = Math.min(a.col, b.col);
  const c2 = Math.max(a.col, b.col);
  const r1 = Math.min(a.row, b.row);
  const r2 = Math.max(a.row, b.row);
  const out = [];
  for (let c = c1; c <= c2; c++) for (let r = r1; r <= r2; r++) out.push(numToCol(c) + r);
  return out;
}

function tokenize(s) {
  const tokens = [];
  let i = 0;
  const isDigit = (c) => c >= '0' && c <= '9';
  const isAlpha = (c) => (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
  while (i < s.length) {
    const c = s[i];
    if (c === ' ' || c === '\t' || c === '\n') { i++; continue; }
    if (c === '"') {
      let j = i + 1;
      let str = '';
      while (j < s.length && s[j] !== '"') { str += s[j]; j++; }
      tokens.push({ t: 'str', v: str });
      i = j + 1;
      continue;
    }
    if (isDigit(c) || (c === '.' && isDigit(s[i + 1]))) {
      let j = i;
      let num = '';
      while (j < s.length && (isDigit(s[j]) || s[j] === '.')) { num += s[j]; j++; }
      tokens.push({ t: 'num', v: Number(num) });
      i = j;
      continue;
    }
    if (isAlpha(c)) {
      let j = i;
      let id = '';
      while (j < s.length && (isAlpha(s[j]) || isDigit(s[j]))) { id += s[j]; j++; }
      tokens.push({ t: 'id', v: id });
      i = j;
      continue;
    }
    const two = s.slice(i, i + 2);
    if (two === '>=' || two === '<=' || two === '<>') { tokens.push({ t: 'op', v: two }); i += 2; continue; }
    if ('+-*/&(),:=<>'.includes(c)) { tokens.push({ t: 'op', v: c }); i++; continue; }
    throw sentinelError('#ERROR!');
  }
  return tokens;
}

function parse(input) {
  const toks = tokenize(input);
  let p = 0;
  const peek = () => toks[p];
  const next = () => toks[p++];
  const expect = (v) => { const t = next(); if (!t || t.v !== v) throw sentinelError('#ERROR!'); };
  const isRef = (v) => /^[A-Z]+[0-9]+$/.test(v);

  function parseExpr() { return parseCompare(); }

  function parseCompare() {
    const left = parseConcat();
    const t = peek();
    if (t && t.t === 'op' && ['=', '<>', '>', '<', '>=', '<='].includes(t.v)) {
      next();
      return { type: 'cmp', op: t.v, left, right: parseConcat() };
    }
    return left;
  }

  function parseConcat() {
    let left = parseAdd();
    while (peek() && peek().t === 'op' && peek().v === '&') { next(); left = { type: 'concat', left, right: parseAdd() }; }
    return left;
  }

  function parseAdd() {
    let left = parseMul();
    while (peek() && peek().t === 'op' && (peek().v === '+' || peek().v === '-')) {
      const op = next().v;
      left = { type: 'bin', op, left, right: parseMul() };
    }
    return left;
  }

  function parseMul() {
    let left = parseUnary();
    while (peek() && peek().t === 'op' && (peek().v === '*' || peek().v === '/')) {
      const op = next().v;
      left = { type: 'bin', op, left, right: parseUnary() };
    }
    return left;
  }

  function parseUnary() {
    if (peek() && peek().t === 'op' && peek().v === '-') { next(); return { type: 'neg', expr: parseUnary() }; }
    return parsePrimary();
  }

  function parsePrimary() {
    const t = peek();
    if (!t) throw sentinelError('#ERROR!');
    if (t.t === 'num') { next(); return { type: 'num', value: t.v }; }
    if (t.t === 'str') { next(); return { type: 'str', value: t.v }; }
    if (t.t === 'op' && t.v === '(') { next(); const e = parseExpr(); expect(')'); return e; }
    if (t.t === 'id') {
      next();
      if (peek() && peek().t === 'op' && peek().v === '(') {
        next();
        const args = [];
        if (!(peek() && peek().v === ')')) {
          args.push(parseExpr());
          while (peek() && peek().v === ',') { next(); args.push(parseExpr()); }
        }
        expect(')');
        return { type: 'call', name: t.v.toUpperCase(), args };
      }
      const ref = t.v.toUpperCase();
      if (!isRef(ref)) throw sentinelError('#ERROR!');
      if (peek() && peek().t === 'op' && peek().v === ':') {
        next();
        const t2 = next();
        if (!t2 || t2.t !== 'id' || !isRef(t2.v.toUpperCase())) throw sentinelError('#ERROR!');
        return { type: 'range', from: ref, to: t2.v.toUpperCase() };
      }
      return { type: 'ref', ref };
    }
    throw sentinelError('#ERROR!');
  }

  const ast = parseExpr();
  if (p !== toks.length) throw sentinelError('#ERROR!');
  return ast;
}

function collectRefs(node, set) {
  if (!node) return;
  switch (node.type) {
    case 'ref': set.add(node.ref); break;
    case 'range': for (const r of expandRange(node.from, node.to)) set.add(r); break;
    case 'bin':
    case 'cmp':
    case 'concat': collectRefs(node.left, set); collectRefs(node.right, set); break;
    case 'neg': collectRefs(node.expr, set); break;
    case 'call': node.args.forEach((a) => collectRefs(a, set)); break;
  }
}

function numToStr(n) {
  if (!isFinite(n)) return String(n);
  if (Number.isInteger(n)) return String(n);
  return String(parseFloat(n.toPrecision(12)));
}

export function valueToString(v) {
  if (v === null || v === undefined) return '';
  if (typeof v === 'boolean') return v ? 'TRUE' : 'FALSE';
  if (typeof v === 'number') return numToStr(v);
  if (Array.isArray(v)) return v.map(valueToString).join(',');
  return String(v);
}

function toNumber(v) {
  if (typeof v === 'number') return v;
  if (typeof v === 'boolean') return v ? 1 : 0;
  if (v === '' || v === null || v === undefined) return 0;
  const n = Number(v);
  if (isNaN(n)) throw sentinelError('#ERROR!');
  return n;
}

function toNumberOrNull(v) {
  if (typeof v === 'number') return v;
  if (typeof v === 'boolean') return v ? 1 : 0;
  if (v === '' || v === null || v === undefined) return null;
  const n = Number(v);
  return isNaN(n) ? null : n;
}

function truthy(v) {
  if (typeof v === 'boolean') return v;
  if (typeof v === 'number') return v !== 0;
  return valueToString(v) !== '';
}

export class Engine {
  constructor(callAI) {
    this.cells = new Map();
    this.callAI = callAI;
    this.onUpdate = null;
  }

  notify(addr) {
    if (this.onUpdate) this.onUpdate(addr);
  }

  getValue(ref) {
    const cell = this.cells.get(ref);
    if (!cell) return '';
    if (cell.state === 'error') throw sentinelError(cell.error);
    return cell.value;
  }

  parseCell(cell) {
    const raw = cell.raw ?? '';
    cell.deps = new Set();
    cell.ast = null;
    cell.template = null;
    cell.literalValue = '';
    if (raw === '') { cell.kind = 'literal'; return; }
    if (raw[0] === '=') {
      const aiMatch = raw.match(/^=AI\(([\s\S]*)\)$/i);
      if (aiMatch) {
        cell.kind = 'ai';
        let tpl = aiMatch[1].trim();
        if (tpl.length >= 2 && tpl[0] === '"' && tpl[tpl.length - 1] === '"') tpl = tpl.slice(1, -1);
        cell.template = tpl;
        let m;
        const braceRe = /\{([A-Za-z]+[0-9]+)\}/g;
        while ((m = braceRe.exec(tpl))) cell.deps.add(m[1].toUpperCase());
        const bareRe = /\b([A-Za-z]{1,2}[0-9]{1,3})\b/g;
        while ((m = bareRe.exec(tpl))) cell.deps.add(m[1].toUpperCase());
        return;
      }
      cell.kind = 'formula';
      try {
        cell.ast = parse(raw.slice(1));
        collectRefs(cell.ast, cell.deps);
      } catch {
        cell.ast = { type: 'error', sentinel: '#ERROR!' };
      }
      return;
    }
    cell.kind = 'literal';
    const num = Number(raw);
    cell.literalValue = raw.trim() !== '' && !isNaN(num) ? num : raw;
  }

  buildDependents() {
    const dependents = new Map();
    for (const [addr, cell] of this.cells) {
      for (const d of cell.deps || []) {
        if (!dependents.has(d)) dependents.set(d, []);
        dependents.get(d).push(addr);
      }
    }
    return dependents;
  }

  collectDirty(addr) {
    const dependents = this.buildDependents();
    const dirty = new Set();
    const stack = [addr];
    while (stack.length) {
      const a = stack.pop();
      if (dirty.has(a)) continue;
      dirty.add(a);
      for (const d of dependents.get(a) || []) stack.push(d);
    }
    return dirty;
  }

  setRaw(addr, raw) {
    let cell = this.cells.get(addr);
    if (!cell) { cell = {}; this.cells.set(addr, cell); }
    cell.raw = raw;
    this.parseCell(cell);
    return this.recompute(this.collectDirty(addr));
  }

  loadAll(sheet) {
    for (const [addr, raw] of Object.entries(sheet)) {
      let cell = this.cells.get(addr);
      if (!cell) { cell = {}; this.cells.set(addr, cell); }
      cell.raw = raw;
      this.parseCell(cell);
    }
    return this.recomputeAll();
  }

  recomputeAll() {
    return this.recompute(new Set(this.cells.keys()));
  }

  async recompute(dirty) {
    const inDeg = new Map();
    const dependents = new Map();
    for (const a of dirty) {
      const cell = this.cells.get(a);
      const deps = cell ? [...cell.deps].filter((d) => dirty.has(d)) : [];
      inDeg.set(a, deps.length);
      for (const d of deps) {
        if (!dependents.has(d)) dependents.set(d, []);
        dependents.get(d).push(a);
      }
    }
    const processed = new Set();
    let frontier = [...dirty].filter((a) => inDeg.get(a) === 0);
    while (frontier.length) {
      await Promise.all(frontier.map(async (a) => {
        await this.computeCell(a);
        processed.add(a);
        this.notify(a);
      }));
      const next = [];
      for (const a of frontier) {
        for (const dep of dependents.get(a) || []) {
          inDeg.set(dep, inDeg.get(dep) - 1);
          if (inDeg.get(dep) === 0) next.push(dep);
        }
      }
      frontier = next;
    }
    for (const a of dirty) {
      if (!processed.has(a)) {
        const cell = this.cells.get(a);
        if (cell) {
          cell.state = 'error';
          cell.error = '#CYCLE!';
          cell.value = '#CYCLE!';
          this.notify(a);
        }
      }
    }
  }

  async computeCell(addr) {
    const cell = this.cells.get(addr);
    if (!cell) return;
    cell.error = null;
    cell.detail = null;
    if (cell.kind === 'literal') {
      cell.value = cell.literalValue;
      cell.state = 'idle';
      return;
    }
    if (cell.kind === 'formula') {
      try {
        cell.value = this.evalNode(cell.ast);
        cell.state = 'idle';
      } catch (e) {
        cell.state = 'error';
        cell.error = e.sentinel || '#ERROR!';
        cell.value = cell.error;
      }
      return;
    }
    if (cell.kind === 'ai') {
      try {
        const prompt = this.buildPrompt(cell);
        cell.state = 'computing';
        this.notify(addr);
        cell.value = await this.callAI(prompt);
        cell.state = 'idle';
      } catch (e) {
        cell.state = 'error';
        cell.error = '#AI_ERROR!';
        cell.detail = String(e.message || e);
        cell.value = '#AI_ERROR!';
      }
    }
  }

  refText(ref) {
    const c = this.cells.get(ref.toUpperCase());
    if (c && c.state === 'error') return c.error;
    return valueToString(c ? c.value : '');
  }

  buildPrompt(cell) {
    let out = cell.template.replace(/\{([A-Za-z]+[0-9]+)\}/g, (_, ref) => this.refText(ref));
    out = out.replace(/\b([A-Za-z]{1,2}[0-9]{1,3})\b/g, (_, ref) => this.refText(ref));
    return out;
  }

  evalNode(node) {
    switch (node.type) {
      case 'num': return node.value;
      case 'str': return node.value;
      case 'ref': return this.getValue(node.ref);
      case 'range': return expandRange(node.from, node.to).map((r) => this.getValue(r));
      case 'neg': return -toNumber(this.evalNode(node.expr));
      case 'bin': return this.evalBin(node);
      case 'cmp': return this.evalCmp(node);
      case 'concat': return valueToString(this.evalNode(node.left)) + valueToString(this.evalNode(node.right));
      case 'call': return this.evalCall(node);
      case 'error': throw sentinelError(node.sentinel);
    }
    throw sentinelError('#ERROR!');
  }

  evalBin(node) {
    const a = toNumber(this.evalNode(node.left));
    const b = toNumber(this.evalNode(node.right));
    switch (node.op) {
      case '+': return a + b;
      case '-': return a - b;
      case '*': return a * b;
      case '/': if (b === 0) throw sentinelError('#ERROR!'); return a / b;
    }
    throw sentinelError('#ERROR!');
  }

  evalCmp(node) {
    const a = this.evalNode(node.left);
    const b = this.evalNode(node.right);
    const pair = typeof a === 'number' && typeof b === 'number' ? [a, b] : [valueToString(a), valueToString(b)];
    switch (node.op) {
      case '=': return pair[0] === pair[1];
      case '<>': return pair[0] !== pair[1];
      case '>': return pair[0] > pair[1];
      case '<': return pair[0] < pair[1];
      case '>=': return pair[0] >= pair[1];
      case '<=': return pair[0] <= pair[1];
    }
    throw sentinelError('#ERROR!');
  }

  evalCall(node) {
    const flat = () => {
      const out = [];
      for (const arg of node.args) {
        const v = this.evalNode(arg);
        if (Array.isArray(v)) out.push(...v);
        else out.push(v);
      }
      return out;
    };
    const nums = () => flat().map(toNumberOrNull).filter((n) => n !== null);
    switch (node.name) {
      case 'SUM': return nums().reduce((s, n) => s + n, 0);
      case 'AVERAGE': { const a = nums(); if (!a.length) throw sentinelError('#ERROR!'); return a.reduce((s, n) => s + n, 0) / a.length; }
      case 'MIN': { const a = nums(); if (!a.length) throw sentinelError('#ERROR!'); return Math.min(...a); }
      case 'MAX': { const a = nums(); if (!a.length) throw sentinelError('#ERROR!'); return Math.max(...a); }
      case 'COUNT': return nums().length;
      case 'CONCAT': return flat().map(valueToString).join('');
      case 'ROUND': {
        const n = toNumber(this.evalNode(node.args[0]));
        const d = node.args.length > 1 ? toNumber(this.evalNode(node.args[1])) : 0;
        const f = Math.pow(10, d);
        return Math.round(n * f) / f;
      }
      case 'IF': {
        if (node.args.length < 2) throw sentinelError('#ERROR!');
        return truthy(this.evalNode(node.args[0]))
          ? this.evalNode(node.args[1])
          : node.args.length > 2 ? this.evalNode(node.args[2]) : false;
      }
      default: throw sentinelError('#ERROR!');
    }
  }
}
