import { describe, it, expect, beforeEach } from 'vitest';
import { BattleComponent } from './battle.component';
import { App } from './app';

describe('BattleComponent - Battle Mechanics', () => {
  let battle: BattleComponent;

  beforeEach(() => {
    battle = Object.create(BattleComponent.prototype);
  });

  it('calcDamage produces higher damage with high attack vs low defense', () => {
    const calcDamage = (BattleComponent.prototype as any).calcDamage;
    const highAtk: number[] = [];
    const lowAtk: number[] = [];
    for (let i = 0; i < 200; i++) {
      highAtk.push(calcDamage(150, 30));
      lowAtk.push(calcDamage(30, 150));
    }
    const avgHigh = highAtk.reduce((a, b) => a + b, 0) / highAtk.length;
    const avgLow = lowAtk.reduce((a, b) => a + b, 0) / lowAtk.length;
    expect(avgHigh).toBeGreaterThan(avgLow);
  });

  it('capitalize handles single character strings', () => {
    expect(battle.capitalize('a')).toBe('A');
    expect(battle.capitalize('Z')).toBe('Z');
  });

  it('formatId handles large pokemon ids', () => {
    expect(battle.formatId(1000)).toBe('#1000');
    expect(battle.formatId(9999)).toBe('#9999');
  });

  it('hpBarColor boundary at exactly 50 returns orange', () => {
    expect(battle.hpBarColor(50)).toBe('#ff9800');
  });

  it('hpBarColor boundary at exactly 20 returns red', () => {
    expect(battle.hpBarColor(20)).toBe('#f44336');
  });

  it('getTypeColor covers all 18 types with unique colors', () => {
    const types = [
      'fire', 'water', 'grass', 'electric', 'ice', 'fighting',
      'poison', 'ground', 'flying', 'psychic', 'bug', 'rock',
      'ghost', 'dragon', 'dark', 'steel', 'fairy', 'normal',
    ];
    const colors = types.map(t => battle.getTypeColor(t));
    const unique = new Set(colors);
    expect(unique.size).toBe(18);
  });
});

describe('App (Pokedex) - Extra Coverage', () => {
  let app: App;

  beforeEach(() => {
    app = Object.create(App.prototype);
  });

  it('formatId pads single digit ids correctly', () => {
    expect(app.formatId(1)).toBe('#001');
    expect(app.formatId(9)).toBe('#009');
  });

  it('capitalize works on multi-word pokemon names', () => {
    expect(app.capitalize('bulbasaur')).toBe('Bulbasaur');
    expect(app.capitalize('mewtwo')).toBe('Mewtwo');
  });

  it('getStatBarColor boundary at exactly 60 returns orange', () => {
    expect(app.getStatBarColor(60)).toBe('#ff9800');
    expect(app.getStatBarColor(99)).toBe('#ff9800');
  });

  it('getStatBarWidth returns proportional values for typical stats', () => {
    const w50 = parseFloat(app.getStatBarWidth(50));
    const w100 = parseFloat(app.getStatBarWidth(100));
    const w200 = parseFloat(app.getStatBarWidth(200));
    expect(w50).toBeLessThan(w100);
    expect(w100).toBeLessThan(w200);
    expect(w200).toBeLessThanOrEqual(100);
  });
});