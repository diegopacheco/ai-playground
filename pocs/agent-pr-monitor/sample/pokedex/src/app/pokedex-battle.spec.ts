import { TestBed } from '@angular/core/testing';
import { App } from './app';
import { BattleComponent } from './battle.component';
import { Pokemon } from './pokemon.service';

const pikachu: Pokemon = {
  id: 25, name: 'pikachu', types: ['electric'], sprite: 'pikachu.png',
  hp: 35, attack: 55, defense: 40, speed: 90, height: 4, weight: 60,
};

const charizard: Pokemon = {
  id: 6, name: 'charizard', types: ['fire', 'flying'], sprite: 'charizard.png',
  hp: 78, attack: 84, defense: 78, speed: 100, height: 17, weight: 905,
};

const bulbasaur: Pokemon = {
  id: 1, name: 'bulbasaur', types: ['grass', 'poison'], sprite: 'bulbasaur.png',
  hp: 45, attack: 49, defense: 49, speed: 45, height: 7, weight: 69,
};

describe('App Pokedex', () => {
  let app: App;

  beforeEach(async () => {
    await TestBed.configureTestingModule({ imports: [App] }).compileComponents();
    app = TestBed.createComponent(App).componentInstance;
  });

  it('should return correct type color for fire', () => {
    expect(app.getTypeColor('fire')).toBe('#F08030');
  });

  it('should return fallback color for unknown type', () => {
    expect(app.getTypeColor('shadow')).toBe('#888');
  });

  it('should format pokemon id with leading zeros', () => {
    expect(app.formatId(1)).toBe('#001');
    expect(app.formatId(25)).toBe('#025');
    expect(app.formatId(150)).toBe('#150');
  });

  it('should capitalize pokemon name', () => {
    expect(app.capitalize('pikachu')).toBe('Pikachu');
  });

  it('should compute stat bar width as percentage of 255', () => {
    expect(app.getStatBarWidth(255)).toBe('100%');
    expect(app.getStatBarWidth(0)).toBe('0%');
    expect(app.getStatBarWidth(127)).toContain('%');
  });

  it('should return green for high stats, orange for mid, red for low', () => {
    expect(app.getStatBarColor(100)).toBe('#4caf50');
    expect(app.getStatBarColor(60)).toBe('#ff9800');
    expect(app.getStatBarColor(30)).toBe('#f44336');
  });

  it('should select and deselect pokemon', () => {
    app.selectPokemon(pikachu);
    expect(app.selectedPokemon()).toEqual(pikachu);
    app.closeDetail();
    expect(app.selectedPokemon()).toBeNull();
  });
});

describe('BattleComponent', () => {
  let battle: BattleComponent;

  beforeEach(async () => {
    await TestBed.configureTestingModule({ imports: [BattleComponent] }).compileComponents();
    battle = TestBed.createComponent(BattleComponent).componentInstance;
  });

  it('should select player then enemy pokemon sequentially', () => {
    battle.selectForBattle(pikachu);
    expect(battle.playerPokemon()).toEqual(pikachu);
    expect(battle.selecting()).toBe('enemy');

    battle.selectForBattle(charizard);
    expect(battle.enemyPokemon()).toEqual(charizard);
    expect(battle.selecting()).toBeNull();
  });

  it('should initialize battle with correct HP values', () => {
    battle.selectForBattle(pikachu);
    battle.selectForBattle(charizard);
    battle.startBattle();

    expect(battle.battleActive()).toBe(true);
    expect(battle.playerHp()).toBe(pikachu.hp);
    expect(battle.enemyHp()).toBe(charizard.hp);
    expect(battle.battleLog().length).toBe(1);
    expect(battle.battleLog()[0].type).toBe('info');
  });

  it('should reset battle to initial state', () => {
    battle.selectForBattle(pikachu);
    battle.selectForBattle(charizard);
    battle.startBattle();
    battle.attack();

    battle.resetBattle();
    expect(battle.playerPokemon()).toBeNull();
    expect(battle.enemyPokemon()).toBeNull();
    expect(battle.battleActive()).toBe(false);
    expect(battle.battleLog()).toEqual([]);
    expect(battle.selecting()).toBe('player');
  });
});
