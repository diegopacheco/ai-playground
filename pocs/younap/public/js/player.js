export const ENDED = 0;
export const PLAYING = 1;
export const PAUSED = 2;
export const BUFFERING = 3;

export function createPlayer(elementId, { onState, onError }) {
  return new Promise(resolve => {
    window.onYouTubeIframeAPIReady = () => {
      const player = new window.YT.Player(elementId, {
        width: '100%',
        height: '100%',
        playerVars: { controls: 0, rel: 0, playsinline: 1, iv_load_policy: 3 },
        events: {
          onReady: () => resolve(player),
          onStateChange: event => onState(event.data),
          onError: event => onError(event.data)
        }
      });
    };
    const script = document.createElement('script');
    script.src = 'https://www.youtube.com/iframe_api';
    document.head.append(script);
  });
}
