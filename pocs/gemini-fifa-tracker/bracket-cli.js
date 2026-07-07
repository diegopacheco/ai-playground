import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { execSync } from 'child_process';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const filePath = path.join(__dirname, 'bracket.json');

const UA = 'FIFA2026TrackerPOC/1.0 (https://github.com/diegopacheco/ai-playground; diego.pacheco.it@gmail.com) node-fetch';
const WIKI_PAGE = '2026 FIFA World Cup knockout stage';

const CODE_TO_NAME = {
  PAR: 'Paraguay', FRA: 'France', CAN: 'Canada', MAR: 'Morocco',
  POR: 'Portugal', ESP: 'Spain', USA: 'USA', BEL: 'Belgium',
  BRA: 'Brazil', NOR: 'Norway', MEX: 'Mexico', ENG: 'England',
  ARG: 'Argentina', EGY: 'Egypt', SUI: 'Switzerland', COL: 'Colombia',
  GER: 'Germany', NED: 'Netherlands', CRO: 'Croatia', JPN: 'Japan',
  AUS: 'Australia', SEN: 'Senegal', GHA: 'Ghana', ALG: 'Algeria'
};

const TEAM_ALIASES = {
  'united states': 'USA', 'united states of america': 'USA',
  'the netherlands': 'Netherlands', 'holland': 'Netherlands',
  'côte d\'ivoire': 'Cote dIvoire', 'ivory coast': 'Cote dIvoire'
};

function normalizeTeam(name) {
  if (!name) return '';
  const known = Object.values(CODE_TO_NAME);
  const hit = known.find(n => n.toLowerCase() === String(name).toLowerCase());
  return hit || TEAM_ALIASES[String(name).toLowerCase()] || String(name);
}

function readBracket() {
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
  } catch (error) {
    return null;
  }
}

function writeBracket(data) {
  fs.writeFileSync(filePath, JSON.stringify(data, null, 2), 'utf8');
}

async function fetchWikipediaBracket() {
  const url = 'https://en.wikipedia.org/w/api.php?' + new URLSearchParams({
    format: 'json', action: 'parse', page: WIKI_PAGE, prop: 'wikitext'
  });
  const res = await fetch(url, { headers: { 'User-Agent': UA } });
  if (!res.ok) throw new Error(`Wikipedia API ${res.status}`);
  const json = await res.json();
  const wikitext = json.parse.wikitext['*'];
  const block = wikitext.split('<section begin="Bracket" />')[1].split('<section end="Bracket" />')[0];

  const segmentOf = (start, end) => {
    const from = block.indexOf(`<!--${start}-->`);
    const to = end ? block.indexOf(`<!--${end}-->`) : block.length;
    return block.slice(from, to);
  };

  const parseMatches = segment =>
    segment.split('\n').filter(l => l.startsWith('|')).map(line => {
      let l = line
        .replace(/\{\{#invoke:flag\|fb(?:-rt)?\|([A-Z]{3})\}\}/g, '$1')
        .replace(/\{\{pso\}\}|\{\{aet\}\}/g, '')
        .replace(/<!--.*?-->/g, '')
        .replace(/\[\[([^\]|]*)\|([^\]]*)\]\]/g, '$2')
        .replace(/\[\[([^\]]*)\]\]/g, '$1');
      const cells = l.split('|').map(c => c.trim());
      const parseScore = c => {
        const m = c.match(/^(\d+)(?:\s*\((\d+)\))?$/);
        return m ? { goals: Number(m[1]), pens: m[2] ? Number(m[2]) : null } : null;
      };
      const parseTeam = c => /^[A-Z]{3}$/.test(c) ? (CODE_TO_NAME[c] || c) : '';
      const date = (cells[1].match(/^(\w+ \d+)/) || [])[1] || '';
      const team1 = parseTeam(cells[2]);
      const team2 = parseTeam(cells[4]);
      const s1 = parseScore(cells[3] || '');
      const s2 = parseScore(cells[5] || '');
      let winner = null;
      if (team1 && team2 && s1 && s2) {
        if (s1.goals !== s2.goals) winner = s1.goals > s2.goals ? team1 : team2;
        else if (s1.pens !== null && s2.pens !== null) winner = s1.pens > s2.pens ? team1 : team2;
      }
      return {
        team1, team2,
        score1: s1 ? s1.goals : null, score2: s2 ? s2.goals : null,
        pens1: s1 ? s1.pens : null, pens2: s2 ? s2.pens : null,
        winner, loser: winner ? (winner === team1 ? team2 : team1) : null,
        date
      };
    });

  const r16 = parseMatches(segmentOf('Round of 16', 'Quarterfinals'));
  const qf = parseMatches(segmentOf('Quarterfinals', 'Semifinals'));
  const sf = parseMatches(segmentOf('Semifinals', 'Final'));
  const fin = parseMatches(segmentOf('Final', 'Match for third place'));
  if (r16.length !== 8 || qf.length !== 4 || sf.length !== 2 || fin.length !== 1) {
    throw new Error(`Unexpected bracket shape: ${r16.length}/${qf.length}/${sf.length}/${fin.length}`);
  }
  return { roundOf16: r16, quarterfinals: qf, semifinals: sf, final: fin[0] };
}

function fetchAgyBracket() {
  const today = new Date().toISOString().slice(0, 10);
  const teams = Object.values(CODE_TO_NAME).join(', ');
  const prompt = `Search the web for the real 2026 FIFA World Cup knockout stage results as of ${today}. ` +
    `Reply ONLY with compact JSON, no markdown fences: {"roundOf16":[8 items],"quarterfinals":[4 items],"semifinals":[2 items],"final":{1 item}} ` +
    `where each item is {"team1","team2","score1","score2","winner"}. ` +
    `Use exactly these team names where applicable: ${teams}. ` +
    `Unplayed matches get null score1/score2/winner. Unknown team slots get empty string.`;
  const raw = execSync(`agy --dangerously-skip-permissions --print "${prompt}"`, {
    encoding: 'utf8', timeout: 180000
  }).trim().replace(/^```(json)?|```$/g, '').trim();
  const data = JSON.parse(raw);
  if (!Array.isArray(data.roundOf16) || data.roundOf16.length !== 8) throw new Error('agy returned bad shape');
  const fix = m => ({
    team1: normalizeTeam(m.team1), team2: normalizeTeam(m.team2),
    score1: m.score1 ?? null, score2: m.score2 ?? null,
    pens1: null, pens2: null,
    winner: m.winner ? normalizeTeam(m.winner) : null,
    loser: m.winner ? (normalizeTeam(m.winner) === normalizeTeam(m.team1) ? normalizeTeam(m.team2) : normalizeTeam(m.team1)) : null,
    date: ''
  });
  return {
    roundOf16: data.roundOf16.map(fix),
    quarterfinals: (data.quarterfinals || []).map(fix),
    semifinals: (data.semifinals || []).map(fix),
    final: fix(data.final || {})
  };
}

function withIds(data) {
  data.roundOf16.forEach((m, i) => { m.id = `r16-${i + 1}`; });
  data.quarterfinals.forEach((m, i) => { m.id = `qf-${i + 1}`; });
  data.semifinals.forEach((m, i) => { m.id = `sf-${i + 1}`; });
  data.final.id = 'f-1';
  return data;
}

async function syncBracket() {
  let agyData = null;
  try {
    console.log('Running agy CLI to fetch current results from the web...');
    agyData = fetchAgyBracket();
    console.log('agy returned knockout data.');
  } catch (e) {
    console.log(`agy fetch failed: ${e.message}`);
  }

  let wikiData = null;
  try {
    console.log('Verifying against Wikipedia live knockout page...');
    wikiData = await fetchWikipediaBracket();
    console.log('Wikipedia data parsed.');
  } catch (e) {
    console.log(`Wikipedia fetch failed: ${e.message}`);
  }

  if (!wikiData && !agyData) {
    console.error('No internet source available. Keeping existing bracket.json untouched.');
    process.exit(1);
  }

  if (wikiData && agyData) {
    wikiData.roundOf16.forEach(m => {
      const pair = [m.team1, m.team2].sort().join();
      const a = agyData.roundOf16.find(x => [x.team1, x.team2].sort().join() === pair);
      if (a && m.winner && a.winner && a.winner !== m.winner) {
        console.log(`Disagreement on ${m.team1} vs ${m.team2}: agy said ${a.winner}, Wikipedia says ${m.winner}. Using Wikipedia.`);
      }
    });
  }

  const data = withIds(wikiData || agyData);
  data.source = wikiData ? 'wikipedia (agy cross-checked)' : 'agy web search';
  data.syncedAt = new Date().toISOString();
  writeBracket(data);
  console.log(`Bracket synced from real web data (source: ${data.source}).`);
  showStatus();
}

function showStatus() {
  const data = readBracket();
  if (!data) {
    console.log('No bracket.json yet. Run with --sync first.');
    return;
  }
  const fmt = m => {
    const score = m.score1 !== null && m.score1 !== undefined && m.score2 !== null
      ? ` ${m.score1}${m.pens1 !== null && m.pens1 !== undefined ? ` (${m.pens1})` : ''}-${m.score2}${m.pens2 !== null && m.pens2 !== undefined ? ` (${m.pens2})` : ''}`
      : '';
    return `${m.team1 || 'TBD'} vs ${m.team2 || 'TBD'}${score} -> Winner: ${m.winner || 'PENDING'}`;
  };
  console.log('--- ROUND OF 16 ---');
  data.roundOf16.forEach(m => console.log(fmt(m)));
  console.log('--- QUARTERFINALS ---');
  data.quarterfinals.forEach(m => console.log(fmt(m)));
  console.log('--- SEMIFINALS ---');
  data.semifinals.forEach(m => console.log(fmt(m)));
  console.log('--- FINAL ---');
  console.log(fmt(data.final));
}

const args = process.argv.slice(2);
if (args.includes('--sync') || args.includes('--update') || args.includes('--reset')) {
  syncBracket();
} else {
  showStatus();
}
