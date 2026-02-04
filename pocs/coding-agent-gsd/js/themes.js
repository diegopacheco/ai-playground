const THEMES = {
  classic: {
    name: 'Classic',
    colors: {
      background: '#0f0f1a',
      grid: '#2a2a4a',
      sidebar: '#16162a',
      I: '#00f0f0',
      O: '#f0f000',
      T: '#a000f0',
      S: '#00f000',
      Z: '#f00000',
      J: '#0000f0',
      L: '#f0a000'
    }
  },
  neon: {
    name: 'Neon',
    colors: {
      background: '#000000',
      grid: '#00ff00',
      sidebar: '#001100',
      I: '#00ffff',
      O: '#ffff00',
      T: '#ff00ff',
      S: '#00ff00',
      Z: '#ff0000',
      J: '#0000ff',
      L: '#ff8800'
    }
  },
  retro: {
    name: 'Retro',
    colors: {
      background: '#f4e4c1',
      grid: '#8b7355',
      sidebar: '#d4c4a1',
      I: '#5f9ea0',
      O: '#daa520',
      T: '#9370db',
      S: '#3cb371',
      Z: '#cd5c5c',
      J: '#4682b4',
      L: '#d2691e'
    }
  },
  minimalist: {
    name: 'Minimalist',
    colors: {
      background: '#f5f5f5',
      grid: '#d0d0d0',
      sidebar: '#e8e8e8',
      I: '#87ceeb',
      O: '#f0e68c',
      T: '#dda0dd',
      S: '#90ee90',
      Z: '#ffb6c1',
      J: '#add8e6',
      L: '#ffa07a'
    }
  },
  highcontrast: {
    name: 'High Contrast',
    colors: {
      background: '#000000',
      grid: '#ffffff',
      sidebar: '#1a1a1a',
      I: '#00ffff',
      O: '#ffff00',
      T: '#ff00ff',
      S: '#00ff00',
      Z: '#ff0000',
      J: '#0080ff',
      L: '#ff8000'
    }
  }
};

const THEME_ORDER = ['classic', 'neon', 'retro', 'minimalist', 'highcontrast'];

let currentTheme = THEMES.classic;

function applyTheme(themeName) {
  if (THEMES[themeName]) {
    currentTheme = THEMES[themeName];
  }
}
