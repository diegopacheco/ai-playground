const { spawn } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');

function readConfig() {
  const config = JSON.parse(fs.readFileSync(path.join(__dirname, 'config.json'), 'utf8'));
  const port = fs.readFileSync(path.join(config.root, 'scripts', 'ports.env'), 'utf8').match(/FRONTEND=(\d+)/)[1];
  return { ...config, url: `http://127.0.0.1:${port}/`, port };
}

function run(config, command, args, onLine) {
  return new Promise(resolve => {
    const child = spawn(command, args, { cwd: config.root, env: { ...process.env, PATH: config.path } });
    const forward = chunk => String(chunk).split('\n').filter(Boolean).forEach(onLine);
    child.stdout.on('data', forward);
    child.stderr.on('data', forward);
    child.on('error', error => { onLine(error.message); resolve(1); });
    child.on('close', code => resolve(code ?? 1));
  });
}

async function reachable(url) {
  try {
    const response = await fetch(url, { signal: AbortSignal.timeout(3000) });
    return response.ok;
  } catch {
    return false;
  }
}

async function startAll(config, report) {
  report('runtime', 'loading', 'Looking for Bun');
  if (await run(config, '/bin/sh', ['-c', 'command -v bun'], line => report('runtime', 'loading', line)) !== 0) {
    report('runtime', 'failed', 'Bun is not installed or not on PATH');
    return false;
  }
  report('runtime', 'ready', 'Bun ready');
  if (fs.existsSync(path.join(config.root, 'node_modules', 'vite'))) report('dependencies', 'ready', 'Dependencies installed');
  else {
    report('dependencies', 'loading', 'Running scripts/setup.sh');
    if (await run(config, path.join(config.root, 'scripts', 'setup.sh'), [], line => report('dependencies', 'loading', line)) !== 0) {
      report('dependencies', 'failed', 'scripts/setup.sh failed');
      return false;
    }
    report('dependencies', 'ready', 'Dependencies installed');
  }
  report('frontend', 'loading', 'Running scripts/start-all.sh');
  if (await run(config, path.join(config.root, 'scripts', 'start-all.sh'), [], line => report('frontend', 'loading', line)) !== 0) {
    report('frontend', 'failed', 'scripts/start-all.sh failed');
    return false;
  }
  report('frontend', 'ready', `Vite ready on port ${config.port}`);
  report('game', 'loading', 'Opening the castle gates');
  if (!await reachable(config.url)) {
    report('game', 'failed', `${config.url} did not respond`);
    return false;
  }
  report('game', 'ready', 'Game ready');
  return true;
}

function stopAll(config) {
  return run(config, path.join(config.root, 'scripts', 'stop-all.sh'), [], line => console.log(line));
}

module.exports = { readConfig, startAll, stopAll };
