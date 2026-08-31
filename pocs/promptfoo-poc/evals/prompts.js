const { render } = require('../src/prompt-loader');

module.exports = {
  classifyTerse: ({ vars }) => render('classify-ticket-terse', vars),
  classifyGuided: ({ vars }) => render('classify-ticket-guided', vars),
  extractFields: ({ vars }) => render('extract-ticket-fields', vars),
};
