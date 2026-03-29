import { TestBed } from '@angular/core/testing';
import { App } from './app';
import { PokemonService, Pokemon } from './pokemon.service';

const fakePokemon: Pokemon = {
  id: 25, name: 'pikachu', types: ['electric'],
  sprite: 'pikachu.png', hp: 35, attack: 55,
  defense: 40, speed: 90, height: 4, weight: 60,
};

const fakeCharizard: Pokemon = {
  id: 6, name: 'charizard', types: ['fire', 'flying'],
  sprite: 'charizard.png', hp: 78, attack: 84,
  defense: 78, speed: 100, height: 17, weight: 905,
};

class MockPokemonService {
  async fetchPokemonList(): Promise<Pokemon[]> {
    return [fakePokemon, fakeCharizard];
  }
  async fetchPokemonById(id: number): Promise<Pokemon> {
    return fakePokemon;
  }
  async searchPokemon(name: string): Promise<Pokemon | null> {
    if (name === 'pikachu') return fakePokemon;
    return null;
  }
}

describe('App', () => {
  let app: App;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [{ provide: PokemonService, useClass: MockPokemonService }],
    }).compileComponents();
    const fixture = TestBed.createComponent(App);
    app = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create the app', () => {
    expect(app).toBeTruthy();
  });

  it('should load pokemon list on init', () => {
    expect(app.pokemonList().length).toBe(2);
    expect(app.pokemonList()[0].name).toBe('pikachu');
  });

  it('should select and close pokemon detail', () => {
    app.selectPokemon(fakePokemon);
    expect(app.selectedPokemon()?.name).toBe('pikachu');
    app.closeDetail();
    expect(app.selectedPokemon()).toBeNull();
  });

  it('should filter by name in search', () => {
    app.searchTerm.set('pika');
    expect(app.filteredList().length).toBe(1);
    expect(app.filteredList()[0].name).toBe('pikachu');
  });

  it('should filter by type in search', () => {
    app.searchTerm.set('fire');
    expect(app.filteredList().length).toBe(1);
    expect(app.filteredList()[0].name).toBe('charizard');
  });

  it('should return all pokemon when search term is empty', () => {
    app.searchTerm.set('');
    expect(app.filteredList().length).toBe(2);
  });

  it('should format pokemon id with leading zeros', () => {
    expect(app.formatId(1)).toBe('#001');
    expect(app.formatId(25)).toBe('#025');
    expect(app.formatId(150)).toBe('#150');
  });

  it('should return correct type colors', () => {
    expect(app.getTypeColor('fire')).toBe('#F08030');
    expect(app.getTypeColor('water')).toBe('#6890F0');
    expect(app.getTypeColor('unknown')).toBe('#888');
  });

  it('should compute stat bar width and color', () => {
    expect(app.getStatBarWidth(255)).toBe('100%');
    expect(app.getStatBarWidth(127)).toContain('49.');
    expect(app.getStatBarColor(120)).toBe('#4caf50');
    expect(app.getStatBarColor(80)).toBe('#ff9800');
    expect(app.getStatBarColor(30)).toBe('#f44336');
  });

  it('should capitalize strings', () => {
    expect(app.capitalize('pikachu')).toBe('Pikachu');
    expect(app.capitalize('charizard')).toBe('Charizard');
  });
});
