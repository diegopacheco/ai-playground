import { describe, it, expect } from 'vitest';

function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function formatId(id: number): string {
  return '#' + id.toString().padStart(3, '0');
}

function getTypeColor(type: string): string {
  const colors: Record<string, string> = {
    fire: '#F08030', water: '#6890F0', grass: '#78C850', electric: '#F8D030',
    ice: '#98D8D8', fighting: '#C03028', poison: '#A040A0', ground: '#E0C068',
    flying: '#A890F0', psychic: '#F85888', bug: '#A8B820', rock: '#B8A038',
    ghost: '#705898', dragon: '#7038F8', dark: '#705848', steel: '#B8B8D0',
    fairy: '#EE99AC', normal: '#A8A878',
  };
  return colors[type] || '#888';
}

function hpBarColor(percent: number): string {
  if (percent > 50) return '#4caf50';
  if (percent > 20) return '#ff9800';
  return '#f44336';
}

function calcDamage(attack: number, defense: number): number {
  const base = Math.max(1, Math.floor((attack * 2) / Math.max(1, defense) * 10));
  const variance = 0.85 + Math.random() * 0.3;
  return Math.max(1, Math.floor(base * variance));
}

function getStatBarWidth(value: number): string {
  return Math.min(value / 255 * 100, 100) + '%';
}

function getStatBarColor(value: number): string {
  if (value >= 100) return '#4caf50';
  if (value >= 60) return '#ff9800';
  return '#f44336';
}

function hpPercent(currentHp: number, maxHp: number): number {
  return Math.max(0, (currentHp / maxHp) * 100);
}

describe('Battle - capitalize', () => {
  it('uppercases the first letter of pokemon names', () => {
    expect(capitalize('pikachu')).toBe('Pikachu');
    expect(capitalize('charizard')).toBe('Charizard');
    expect(capitalize('')).toBe('');
  });
});

describe('Battle - formatId', () => {
  it('pads pokemon id to 3 digits with hash prefix', () => {
    expect(formatId(1)).toBe('#001');
    expect(formatId(25)).toBe('#025');
    expect(formatId(150)).toBe('#150');
    expect(formatId(1000)).toBe('#1000');
  });
});

describe('Battle - getTypeColor', () => {
  it('returns correct hex color for known pokemon types', () => {
    expect(getTypeColor('fire')).toBe('#F08030');
    expect(getTypeColor('water')).toBe('#6890F0');
    expect(getTypeColor('grass')).toBe('#78C850');
    expect(getTypeColor('electric')).toBe('#F8D030');
    expect(getTypeColor('dragon')).toBe('#7038F8');
  });

  it('returns fallback color for unknown types', () => {
    expect(getTypeColor('cosmic')).toBe('#888');
    expect(getTypeColor('')).toBe('#888');
  });
});

describe('Battle - hpBarColor', () => {
  it('returns green above 50%, orange 20-50%, red at or below 20%', () => {
    expect(hpBarColor(80)).toBe('#4caf50');
    expect(hpBarColor(51)).toBe('#4caf50');
    expect(hpBarColor(50)).toBe('#ff9800');
    expect(hpBarColor(21)).toBe('#ff9800');
    expect(hpBarColor(20)).toBe('#f44336');
    expect(hpBarColor(5)).toBe('#f44336');
    expect(hpBarColor(0)).toBe('#f44336');
  });
});

describe('Battle - calcDamage', () => {
  it('always returns at least 1 even with very low attack', () => {
    for (let i = 0; i < 50; i++) {
      expect(calcDamage(1, 999)).toBeGreaterThanOrEqual(1);
    }
  });

  it('produces higher average damage when attack exceeds defense', () => {
    const results: number[] = [];
    for (let i = 0; i < 100; i++) {
      results.push(calcDamage(100, 50));
    }
    const avg = results.reduce((a, b) => a + b, 0) / results.length;
    expect(avg).toBeGreaterThan(10);
  });
});

describe('Battle - hpPercent', () => {
  it('calculates correct HP percentage', () => {
    expect(hpPercent(35, 35)).toBe(100);
    expect(hpPercent(0, 35)).toBe(0);
    expect(hpPercent(17, 35)).toBeCloseTo(48.57, 1);
  });

  it('clamps to 0 for negative HP', () => {
    expect(hpPercent(-5, 35)).toBe(0);
  });
});

describe('Pokedex - getTypeColor covers all 18 types', () => {
  it('returns a valid non-fallback hex for every official type', () => {
    const types = [
      'fire', 'water', 'grass', 'electric', 'ice', 'fighting',
      'poison', 'ground', 'flying', 'psychic', 'bug', 'rock',
      'ghost', 'dragon', 'dark', 'steel', 'fairy', 'normal',
    ];
    for (const t of types) {
      const color = getTypeColor(t);
      expect(color).toMatch(/^#[A-Fa-f0-9]{6}$/);
      expect(color).not.toBe('#888');
    }
  });
});

describe('Pokedex - getStatBarWidth', () => {
  it('scales stat value to percentage clamped at 100%', () => {
    expect(getStatBarWidth(255)).toBe('100%');
    expect(getStatBarWidth(0)).toBe('0%');
    expect(getStatBarWidth(300)).toBe('100%');
    const mid = parseFloat(getStatBarWidth(128));
    expect(mid).toBeGreaterThan(49);
    expect(mid).toBeLessThan(52);
  });
});

describe('Pokedex - getStatBarColor', () => {
  it('returns green for high stats, orange for mid, red for low', () => {
    expect(getStatBarColor(100)).toBe('#4caf50');
    expect(getStatBarColor(150)).toBe('#4caf50');
    expect(getStatBarColor(60)).toBe('#ff9800');
    expect(getStatBarColor(80)).toBe('#ff9800');
    expect(getStatBarColor(59)).toBe('#f44336');
    expect(getStatBarColor(10)).toBe('#f44336');
  });
});