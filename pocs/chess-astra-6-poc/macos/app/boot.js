const log = document.getElementById('log');
const actions = document.getElementById('actions');
const labels = { loading: 'Loading', ready: 'Ready', failed: 'Failed' };

window.macApp.onBoot(({ id, status, detail }) => {
  const row = document.querySelector(`[data-id="${id}"]`);
  row.dataset.status = status;
  row.querySelector('small').textContent = `${labels[status]} · ${detail}`;
  log.textContent += `${detail}\n`;
  log.scrollTop = log.scrollHeight;
  if (status === 'failed') {
    document.querySelector('h1').textContent = 'The gates are stuck.';
    actions.hidden = false;
  }
});
document.getElementById('retry').onclick = () => window.macApp.retry();
document.getElementById('quit').onclick = () => window.macApp.quit();
