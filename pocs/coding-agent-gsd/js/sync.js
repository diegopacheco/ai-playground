const channel = new BroadcastChannel('tetris-sync');

function sendMessage(type, payload) {
  channel.postMessage({ type: type, payload: payload });
}

window.addEventListener('beforeunload', function() {
  channel.close();
});

channel.addEventListener('messageerror', function(event) {
  console.error('Sync message error:', event);
});
