import assert from 'node:assert';
import { Engine } from './public/engine.js';

const fakeAI = async (prompt) => `AI:${prompt}`;

async function formulas() {
  const e = new Engine(fakeAI);
  await e.setRaw('A1', '10');
  await e.setRaw('A2', '20');
  await e.setRaw('A3', '=A1+A2');
  assert.strictEqual(e.cells.get('A3').value, 30);

  await e.setRaw('A4', '=SUM(A1:A3)');
  assert.strictEqual(e.cells.get('A4').value, 60);

  await e.setRaw('A1', '15');
  assert.strictEqual(e.cells.get('A3').value, 35);
  assert.strictEqual(e.cells.get('A4').value, 70);

  await e.setRaw('B1', '=A1*2');
  assert.strictEqual(e.cells.get('B1').value, 30);

  await e.setRaw('C1', '=IF(A1>10,"big","small")');
  assert.strictEqual(e.cells.get('C1').value, 'big');

  await e.setRaw('D1', '="Total " & A1');
  assert.strictEqual(e.cells.get('D1').value, 'Total 15');

  await e.setRaw('E1', '=AVERAGE(A1:A2)');
  assert.strictEqual(e.cells.get('E1').value, 17.5);

  await e.setRaw('F1', '=ROUND(10/3,2)');
  assert.strictEqual(e.cells.get('F1').value, 3.33);

  await e.setRaw('G1', '=MAX(A1:A3)');
  assert.strictEqual(e.cells.get('G1').value, 35);

  await e.setRaw('H1', '=sum(A1,A2)');
  assert.strictEqual(e.cells.get('H1').value, 35);
}

async function aiCell() {
  const e = new Engine(fakeAI);
  await e.setRaw('A1', 'mars');
  await e.setRaw('B1', '=AI("describe {A1}")');
  assert.deepStrictEqual([...e.cells.get('B1').deps], ['A1']);
  assert.strictEqual(e.cells.get('B1').value, 'AI:describe mars');

  await e.setRaw('A1', 'venus');
  assert.strictEqual(e.cells.get('B1').value, 'AI:describe venus');

  const u = new Engine(fakeAI);
  await u.setRaw('A1', '110');
  await u.setRaw('A2', '220');
  await u.setRaw('A3', '=ai(sum a1 and a2)');
  assert.deepStrictEqual([...u.cells.get('A3').deps].sort(), ['A1', 'A2']);
  assert.strictEqual(u.cells.get('A3').value, 'AI:sum 110 and 220');

  await u.setRaw('A1', '5');
  assert.strictEqual(u.cells.get('A3').value, 'AI:sum 5 and 220');
}

async function cycles() {
  const e = new Engine(fakeAI);
  await e.setRaw('X1', '=Y1');
  await e.setRaw('Y1', '=X1');
  assert.strictEqual(e.cells.get('X1').error, '#CYCLE!');
  assert.strictEqual(e.cells.get('Y1').error, '#CYCLE!');
}

async function errors() {
  const e = new Engine(fakeAI);
  await e.setRaw('A1', '=1/0');
  assert.strictEqual(e.cells.get('A1').error, '#ERROR!');
  await e.setRaw('A2', '=A1+1');
  assert.strictEqual(e.cells.get('A2').error, '#ERROR!');
}

async function main() {
  await formulas();
  await aiCell();
  await cycles();
  await errors();
  console.log('all engine tests passed');
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
