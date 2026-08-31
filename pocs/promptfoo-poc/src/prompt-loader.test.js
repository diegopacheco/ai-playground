const { test } = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');
const { render } = require('./prompt-loader');

const PROMPTS_DIR = path.join(__dirname, '..', 'prompts');

test('every prompt file renders, so a prompt edit never needs a code edit', () => {
  const names = fs
    .readdirSync(PROMPTS_DIR)
    .filter((f) => f.endsWith('.md') && f !== 'README.md')
    .map((f) => f.replace(/\.md$/, ''));

  assert.ok(names.length > 0, 'no prompts found');
  for (const name of names) {
    const out = render(name, { ticket_body: 'the app crashed', max_summary_words: 15 });
    assert.match(out, /the app crashed/);
    assert.doesNotMatch(out, /\$[a-z]/, `${name} still has an unsubstituted parameter`);
  }
});

test('a missing value fails loud instead of sending "undefined" to the model', () => {
  assert.throws(
    () => render('extract-ticket-fields', { ticket_body: 'x' }),
    /no value for \$max_summary_words/,
  );
});

test('an empty value fails loud, because an empty ticket makes the eval meaningless', () => {
  assert.throws(() => render('classify-ticket-guided', { ticket_body: '' }), /no value for \$ticket_body/);
});

test('the header is the contract: an undeclared parameter is rejected', () => {
  const file = path.join(PROMPTS_DIR, 'undeclared-param.md');
  fs.writeFileSync(file, '---\nname: undeclared-param\nparams: $known\n---\n\n$known and $sneaky\n');
  try {
    assert.throws(() => render('undeclared-param', { known: 'a', sneaky: 'b' }), /\$sneaky is used but not declared/);
  } finally {
    fs.unlinkSync(file);
  }
});

test('a "$" inside a ticket is not expanded again, so ticket text cannot inject parameters', () => {
  const out = render('classify-ticket-guided', { ticket_body: 'you charged me $ticket_body twice' });
  assert.match(out, /you charged me \$ticket_body twice/);
});
