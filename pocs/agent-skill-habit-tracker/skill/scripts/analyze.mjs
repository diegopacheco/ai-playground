import fs from 'fs';
import path from 'path';
import os from 'os';
import { fileURLToPath } from 'url';

const here = path.dirname(fileURLToPath(import.meta.url));
const claudeDir = process.env.CLAUDE_CONFIG_DIR || path.join(os.homedir(), '.claude');
const projectsDir = path.join(claudeDir, 'projects');
const outDir = path.resolve(process.argv[2] || 'habit-report');

const pad = (n) => String(n).padStart(2, '0');
const dayKey = (d) => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;

function walk(dir) {
  const out = [];
  let entries;
  try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch { return out; }
  for (const e of entries) {
    const full = path.join(dir, e.name);
    if (e.isDirectory()) out.push(...walk(full));
    else if (e.isFile() && e.name.endsWith('.jsonl')) out.push(full);
  }
  return out;
}

const files = walk(projectsDir);
if (files.length === 0) {
  console.error(`No session files found under ${projectsDir}`);
  process.exit(2);
}

const days = new Map();
const hours = new Array(24).fill(0);
const weekdays = new Array(7).fill(0);
const tools = new Map();
const projects = new Map();
const models = new Map();
const sessions = new Map();
const themeTokens = new Map();
const stop = new Set(['agent', 'skill', 'poc', 'pocs', 'the', 'and', 'app', 'test', 'new', 'my', 'fun', 'demo', 'sample']);
const intent = { build: 0, fix: 0, refactor: 0, test: 0, explore: 0 };
const buildWords = /\b(build|create|add|implement|generate|make|write|new|set up|setup|draw|render)\b/i;
const fixWords = /\b(fix|bug|error|fail|failing|broken|debug|issue|wrong|crash|not working|doesn'?t work)\b/i;
const refactorWords = /\b(refactor|clean|cleanup|simplify|rename|reorganize|restructure|tidy)\b/i;
const testWords = /\b(test|verify|check|validate|assert|coverage)\b/i;
const exploreWords = /\b(explain|why|how|what|understand|review|read|look at|investigate)\b/i;

let firstTs = null;
let lastTs = null;
let totalUser = 0;
let totalAssistant = 0;
let totalTools = 0;

function bump(map, key, n = 1) { map.set(key, (map.get(key) || 0) + n); }

function textOf(content) {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content.filter((b) => b && b.type === 'text').map((b) => b.text || '').join(' ');
  }
  return '';
}

function classify(text) {
  if (!text) return;
  const t = text.slice(0, 600);
  if (fixWords.test(t)) intent.fix++;
  else if (refactorWords.test(t)) intent.refactor++;
  else if (testWords.test(t)) intent.test++;
  else if (buildWords.test(t)) intent.build++;
  else if (exploreWords.test(t)) intent.explore++;
}

for (const file of files) {
  let raw;
  try { raw = fs.readFileSync(file, 'utf8'); } catch { continue; }
  for (const line of raw.split('\n')) {
    if (!line) continue;
    let o;
    try { o = JSON.parse(line); } catch { continue; }
    const type = o.type;
    if (type !== 'user' && type !== 'assistant') continue;
    const ts = o.timestamp;
    if (!ts) continue;
    const d = new Date(ts);
    if (isNaN(d.getTime())) continue;
    if (!firstTs || d < firstTs) firstTs = d;
    if (!lastTs || d > lastTs) lastTs = d;

    const key = dayKey(d);
    let day = days.get(key);
    if (!day) { day = { count: 0, tools: 0, sessions: new Set() }; days.set(key, day); }
    day.count++;
    hours[d.getHours()]++;
    weekdays[d.getDay()]++;

    const sid = o.sessionId || 'unknown';
    day.sessions.add(sid);
    let sess = sessions.get(sid);
    if (!sess) { sess = { start: d, end: d, msgs: 0, tools: 0, project: null }; sessions.set(sid, sess); }
    if (d < sess.start) sess.start = d;
    if (d > sess.end) sess.end = d;
    sess.msgs++;

    const cwd = o.cwd || '';
    if (cwd) {
      const name = path.basename(cwd) || cwd;
      sess.project = sess.project || name;
      let p = projects.get(name);
      if (!p) { p = { events: 0, sessions: new Set(), first: d, last: d, branches: new Set() }; projects.set(name, p); }
      p.events++;
      p.sessions.add(sid);
      if (d < p.first) p.first = d;
      if (d > p.last) p.last = d;
      if (o.gitBranch) p.branches.add(o.gitBranch);
      for (const tok of name.toLowerCase().split(/[^a-z0-9]+/)) {
        if (tok.length >= 3 && !stop.has(tok) && !/^\d+$/.test(tok)) bump(themeTokens, tok);
      }
    }

    const msg = o.message && typeof o.message === 'object' ? o.message : {};
    if (type === 'assistant') {
      totalAssistant++;
      if (msg.model) bump(models, msg.model);
      const c = msg.content;
      if (Array.isArray(c)) {
        for (const b of c) {
          if (b && b.type === 'tool_use' && b.name) {
            bump(tools, b.name);
            totalTools++;
            day.tools++;
            sess.tools++;
          }
        }
      }
    } else {
      const t = textOf(msg.content);
      if (t && !t.startsWith('<command-') && !t.startsWith('[Request interrupted')) {
        totalUser++;
        classify(t);
      }
    }
  }
}

const now = new Date();
const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());

function streaks() {
  let current = 0;
  const probe = new Date(today);
  while (days.has(dayKey(probe))) { current++; probe.setDate(probe.getDate() - 1); }
  let longest = 0;
  let longestEnd = null;
  const keys = [...days.keys()].sort();
  let run = 0;
  let prev = null;
  for (const k of keys) {
    const d = new Date(k + 'T00:00:00');
    if (prev && (d - prev) === 86400000) run++;
    else run = 1;
    if (run > longest) { longest = run; longestEnd = d; }
    prev = d;
  }
  return { current, longest, longestEnd };
}

const { current: currentStreak, longest: longestStreak, longestEnd } = streaks();

const spanDays = firstTs ? Math.max(1, Math.round((today - new Date(firstTs.getFullYear(), firstTs.getMonth(), firstTs.getDate())) / 86400000) + 1) : 1;
const activeDays = days.size;
const totalSessions = sessions.size;
const totalEvents = [...days.values()].reduce((a, d) => a + d.count, 0);

const peakHour = hours.indexOf(Math.max(...hours));
const nightShare = hours.slice(20).reduce((a, b) => a + b, 0) + hours.slice(0, 3).reduce((a, b) => a + b, 0);
const morningShare = hours.slice(5, 12).reduce((a, b) => a + b, 0);
const totalHourEvents = hours.reduce((a, b) => a + b, 0) || 1;

const wdNames = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
const peakWeekday = weekdays.indexOf(Math.max(...weekdays));
const weekendShare = (weekdays[0] + weekdays[6]) / (weekdays.reduce((a, b) => a + b, 0) || 1);

const topTools = [...tools.entries()].sort((a, b) => b[1] - a[1]);
const readN = (tools.get('Read') || 0) + (tools.get('Grep') || 0) + (tools.get('Glob') || 0);
const writeN = (tools.get('Edit') || 0) + (tools.get('Write') || 0) + (tools.get('NotebookEdit') || 0);

const topProjects = [...projects.entries()]
  .map(([name, p]) => ({ name, events: p.events, sessions: p.sessions.size, branches: p.branches.size, last: p.last }))
  .sort((a, b) => b.events - a.events);

const topThemes = [...themeTokens.entries()].sort((a, b) => b[1] - a[1]).slice(0, 6);

const recentWindow = Math.min(90, spanDays);
const recentCut = new Date(today); recentCut.setDate(recentCut.getDate() - recentWindow);
const recentProjects = [...projects.values()].filter((p) => p.last >= recentCut).length;

const last30 = new Date(today); last30.setDate(last30.getDate() - 30);
const prev30 = new Date(today); prev30.setDate(prev30.getDate() - 60);
let last30Events = 0;
let prev30Events = 0;
for (const [k, d] of days) {
  const dd = new Date(k + 'T00:00:00');
  if (dd >= last30) last30Events += d.count;
  else if (dd >= prev30) prev30Events += d.count;
}
const momentum = (spanDays >= 60 && prev30Events >= 20) ? Math.round(((last30Events - prev30Events) / prev30Events) * 100) : null;

const sessArr = [...sessions.values()];
const avgMsgs = totalSessions ? Math.round(totalEvents / totalSessions) : 0;
const avgTools = totalSessions ? Math.round(totalTools / totalSessions) : 0;
const marathon = sessArr.map((s) => ({ mins: Math.round((s.end - s.start) / 60000), start: s.start, project: s.project }))
  .sort((a, b) => b.mins - a.mins)[0] || { mins: 0 };

const topModel = [...models.entries()].sort((a, b) => b[1] - a[1])[0];
const totalModelEvents = [...models.values()].reduce((a, b) => a + b, 0) || 1;

const fmtDate = (d) => d ? d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }) : '';
const fmtHour = (h) => { const ap = h < 12 ? 'am' : 'pm'; const hh = h % 12 === 0 ? 12 : h % 12; return `${hh}${ap}`; };

const candidates = [];
function add(emoji, title, text, priority) { candidates.push({ emoji, title, text, priority }); }

if (firstTs) add('🗓️', 'Your journey', `You have ${totalEvents.toLocaleString()} agent interactions across ${totalSessions.toLocaleString()} sessions, spanning ${spanDays} days from ${fmtDate(firstTs)} to today.`, 50);

if (currentStreak >= 2) add('🔥', 'On a roll', `You're on a ${currentStreak}-day coding streak right now. Your all-time record is ${longestStreak} days${longestEnd ? `, set around ${fmtDate(longestEnd)}` : ''}.`, currentStreak >= 5 ? 95 : 80);
else add('🔥', 'Longest streak', `Your longest run of consecutive coding days was ${longestStreak} days${longestEnd ? `, around ${fmtDate(longestEnd)}` : ''}.`, 60);

if (nightShare / totalHourEvents > 0.35) add('🦉', 'Night owl', `${Math.round((nightShare / totalHourEvents) * 100)}% of your activity happens late at night or before dawn. Your single busiest hour is ${fmtHour(peakHour)}.`, 90);
else if (morningShare / totalHourEvents > 0.45) add('🌅', 'Early bird', `${Math.round((morningShare / totalHourEvents) * 100)}% of your work happens in the morning. Your busiest hour is ${fmtHour(peakHour)}.`, 90);
else add('⏰', 'Peak hour', `Your most productive hour of the day is ${fmtHour(peakHour)}, with ${hours[peakHour].toLocaleString()} interactions logged.`, 70);

add('📆', 'Power day', `${wdNames[peakWeekday]} is your most active day, carrying ${Math.round((weekdays[peakWeekday] / (weekdays.reduce((a, b) => a + b, 0) || 1)) * 100)}% of all your activity.`, 75);

if (weekendShare > 0.3) add('🏖️', 'Weekend warrior', `${Math.round(weekendShare * 100)}% of your coding happens on weekends — you build for the love of it, not just on the clock.`, 78);

if (topThemes.length && topProjects.length) add('🛠️', 'What you build', `Your work centers on ${topThemes.slice(0, 3).map((t) => t[0]).join(', ')}. Your most active project is "${topProjects[0].name}" with ${topProjects[0].events.toLocaleString()} interactions.`, 92);

if (topTools.length) {
  const share = Math.round((topTools[0][1] / (totalTools || 1)) * 100);
  let ratioText = '';
  if (writeN && readN) {
    const r = readN / writeN;
    if (r >= 1.2) ratioText = ` You run ${r.toFixed(1)} reads for every edit — you look before you leap.`;
    else if (r <= 0.8) ratioText = ` You make ${(writeN / readN).toFixed(1)} edits for every read — you write fast and iterate.`;
    else ratioText = ` Your reads and edits stay roughly balanced.`;
  }
  add('🧰', 'Tool fingerprint', `${topTools[0][0]} is your most-used tool (${share}% of all tool calls).${ratioText}`, 85);
}

const intentTotal = Object.values(intent).reduce((a, b) => a + b, 0) || 1;
const builderPct = Math.round((intent.build / intentTotal) * 100);
const fixerPct = Math.round((intent.fix / intentTotal) * 100);
if (intent.build >= intent.fix) add('🏗️', 'Builder mindset', `${builderPct}% of your prompts start new work versus ${fixerPct}% fixing bugs — you spend more time creating than repairing.`, 82);
else add('🔧', 'Fixer mindset', `${fixerPct}% of your prompts are about fixing and debugging versus ${builderPct}% starting fresh — you keep things running.`, 82);

if (recentProjects >= 3) {
  const perWeek = recentProjects / (recentWindow / 7);
  add('🪁', 'Breadth', `You've worked across ${recentProjects} distinct projects in the last ${recentWindow} days — about ${perWeek >= 2 ? Math.round(perWeek) : perWeek.toFixed(1)} per week. You context-switch a lot.`, 76);
}

add('⚡', 'Session intensity', `A typical session runs ${avgMsgs} messages and ${avgTools} tool calls. Your longest single session lasted ${Math.floor(marathon.mins / 60)}h ${marathon.mins % 60}m${marathon.project ? ` on "${marathon.project}"` : ''}.`, 72);

if (momentum !== null) add(momentum >= 0 ? '📈' : '📉', 'Momentum', `Your activity is ${momentum >= 0 ? 'up' : 'down'} ${Math.abs(momentum)}% over the last 30 days compared with the 30 before it.`, 88);

add('🎯', 'Consistency', `You showed up to code on ${activeDays} of the last ${spanDays} days (${Math.round((activeDays / spanDays) * 100)}%), averaging ${(totalSessions / activeDays).toFixed(1)} sessions per active day.`, 74);

if (topModel) add('🤖', 'Model of choice', `${Math.round((topModel[1] / totalModelEvents) * 100)}% of your assistant turns run on ${topModel[0]}.`, 55);

const insights = candidates.sort((a, b) => b.priority - a.priority).slice(0, 10);

const data = {
  generatedAt: now.toISOString(),
  span: { first: firstTs ? firstTs.toISOString() : null, days: spanDays },
  totals: {
    events: totalEvents,
    sessions: totalSessions,
    activeDays,
    toolCalls: totalTools,
    projects: projects.size,
    userPrompts: totalUser,
    assistantTurns: totalAssistant,
  },
  streak: { current: currentStreak, longest: longestStreak },
  days: Object.fromEntries([...days.entries()].map(([k, v]) => [k, v.count])),
  hours,
  weekdays,
  tools: topTools.slice(0, 12),
  projects: topProjects.slice(0, 12),
  themes: topThemes,
  intent,
  models: [...models.entries()].sort((a, b) => b[1] - a[1]),
  insights,
};

fs.mkdirSync(outDir, { recursive: true });
const template = fs.readFileSync(path.join(here, '..', 'assets', 'template.html'), 'utf8');
const html = template.replace('"__DATA__"', JSON.stringify(data));
fs.writeFileSync(path.join(outDir, 'index.html'), html);
fs.writeFileSync(path.join(outDir, 'data.json'), JSON.stringify(data, null, 2));

console.log(`Scanned ${files.length} session files.`);
console.log(`${totalEvents.toLocaleString()} interactions, ${totalSessions} sessions, ${activeDays} active days across ${spanDays} days.`);
console.log(`Current streak ${currentStreak}d, longest ${longestStreak}d. Busiest hour ${fmtHour(peakHour)}, busiest day ${wdNames[peakWeekday]}.`);
console.log(`Top project: ${topProjects[0] ? topProjects[0].name : 'n/a'}. Generated ${insights.length} insights.`);
console.log(`Report written to ${path.join(outDir, 'index.html')}`);
