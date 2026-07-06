import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { execSync } from 'child_process';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const filePath = path.join(__dirname, 'bracket.json');

const initialData = {
  roundOf16: [
    { id: 'r16-1', team1: 'Argentina', team2: 'Mexico', winner: 'Argentina', loser: 'Mexico' },
    { id: 'r16-2', team1: 'Brazil', team2: 'USA', winner: 'Brazil', loser: 'USA' },
    { id: 'r16-3', team1: 'France', team2: 'Canada', winner: 'France', loser: 'Canada' },
    { id: 'r16-4', team1: 'England', team2: 'Morocco', winner: 'England', loser: 'Morocco' },
    { id: 'r16-5', team1: 'Spain', team2: 'Japan', winner: 'Spain', loser: 'Japan' },
    { id: 'r16-6', team1: 'Portugal', team2: 'South Korea', winner: 'Portugal', loser: 'South Korea' },
    { id: 'r16-7', team1: 'Germany', team2: 'Australia', winner: 'Germany', loser: 'Australia' },
    { id: 'r16-8', team1: 'Italy', team2: 'Saudi Arabia', winner: 'Italy', loser: 'Saudi Arabia' }
  ],
  quarterfinals: [
    { id: 'qf-1', team1: 'Argentina', team2: 'Brazil', winner: null, loser: null },
    { id: 'qf-2', team1: 'France', team2: 'England', winner: null, loser: null },
    { id: 'qf-3', team1: 'Spain', team2: 'Portugal', winner: null, loser: null },
    { id: 'qf-4', team1: 'Germany', team2: 'Italy', winner: null, loser: null }
  ],
  semifinals: [
    { id: 'sf-1', team1: '', team2: '', winner: null, loser: null },
    { id: 'sf-2', team1: '', team2: '', winner: null, loser: null }
  ],
  final: { id: 'f-1', team1: '', team2: '', winner: null, loser: null }
};

function readBracket() {
  try {
    const content = fs.readFileSync(filePath, 'utf8');
    return JSON.parse(content);
  } catch (error) {
    return initialData;
  }
}

function writeBracket(data) {
  fs.writeFileSync(filePath, JSON.stringify(data, null, 2), 'utf8');
}

function getAIWinner(team1, team2) {
  try {
    const prompt = `Which team wins realistically in a football match between ${team1} and ${team2} in the FIFA World Cup 2026? Reply with exactly and only the name of the winning team with no explanation.`;
    const command = `agy --dangerously-skip-permissions --print "${prompt}"`;
    const result = execSync(command, { encoding: 'utf8' }).trim();
    if (result.toLowerCase().includes(team1.toLowerCase())) {
      return team1;
    }
    if (result.toLowerCase().includes(team2.toLowerCase())) {
      return team2;
    }
    return Math.random() < 0.5 ? team1 : team2;
  } catch (error) {
    return Math.random() < 0.5 ? team1 : team2;
  }
}

function resetBracket() {
  writeBracket(initialData);
  console.log('Bracket has been reset to the initial state.');
}

function updateBracket() {
  const data = readBracket();

  for (let i = 0; i < data.roundOf16.length; i++) {
    const match = data.roundOf16[i];
    if (match.winner === null) {
      const winner = getAIWinner(match.team1, match.team2);
      match.winner = winner;
      match.loser = winner === match.team1 ? match.team2 : match.team1;

      const nextMatchIndex = Math.floor(i / 2);
      const isTeam1 = i % 2 === 0;
      if (isTeam1) {
        data.quarterfinals[nextMatchIndex].team1 = winner;
      } else {
        data.quarterfinals[nextMatchIndex].team2 = winner;
      }
      writeBracket(data);
      console.log(`Updated R16 Match ${match.id}: Winner is ${winner}`);
      return;
    }
  }

  for (let i = 0; i < data.quarterfinals.length; i++) {
    const match = data.quarterfinals[i];
    if (match.team1 && match.team2 && match.winner === null) {
      const winner = getAIWinner(match.team1, match.team2);
      match.winner = winner;
      match.loser = winner === match.team1 ? match.team2 : match.team1;

      const nextMatchIndex = Math.floor(i / 2);
      const isTeam1 = i % 2 === 0;
      if (isTeam1) {
        data.semifinals[nextMatchIndex].team1 = winner;
      } else {
        data.semifinals[nextMatchIndex].team2 = winner;
      }
      writeBracket(data);
      console.log(`Updated Quarterfinal Match ${match.id}: Winner is ${winner}`);
      return;
    }
  }

  for (let i = 0; i < data.semifinals.length; i++) {
    const match = data.semifinals[i];
    if (match.team1 && match.team2 && match.winner === null) {
      const winner = getAIWinner(match.team1, match.team2);
      match.winner = winner;
      match.loser = winner === match.team1 ? match.team2 : match.team1;

      if (i === 0) {
        data.final.team1 = winner;
      } else {
        data.final.team2 = winner;
      }
      writeBracket(data);
      console.log(`Updated Semifinal Match ${match.id}: Winner is ${winner}`);
      return;
    }
  }

  const match = data.final;
  if (match.team1 && match.team2 && match.winner === null) {
    const winner = getAIWinner(match.team1, match.team2);
    match.winner = winner;
    match.loser = winner === match.team1 ? match.team2 : match.team1;
    writeBracket(data);
    console.log(`Tournament complete! Champion is ${winner}`);
    return;
  }

  console.log('All bracket matches are already updated.');
}

function showStatus() {
  const data = readBracket();
  console.log('--- ROUND OF 16 ---');
  data.roundOf16.forEach(m => console.log(`${m.team1} vs ${m.team2} -> Winner: ${m.winner || 'PENDING'}`));
  console.log('--- QUARTERFINALS ---');
  data.quarterfinals.forEach(m => console.log(`${m.team1 || 'TBD'} vs ${m.team2 || 'TBD'} -> Winner: ${m.winner || 'PENDING'}`));
  console.log('--- SEMIFINALS ---');
  data.semifinals.forEach(m => console.log(`${m.team1 || 'TBD'} vs ${m.team2 || 'TBD'} -> Winner: ${m.winner || 'PENDING'}`));
  console.log('--- FINAL ---');
  console.log(`${data.final.team1 || 'TBD'} vs ${data.final.team2 || 'TBD'} -> Champion: ${data.final.winner || 'PENDING'}`);
}

const args = process.argv.slice(2);
if (args.includes('--reset')) {
  resetBracket();
} else if (args.includes('--update')) {
  updateBracket();
} else {
  showStatus();
}
