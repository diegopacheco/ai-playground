import { Component, signal, computed } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { PokemonService, Pokemon } from './pokemon.service';
import { BattleComponent } from './battle.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule, BattleComponent],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App {
  activeTab = signal<'pokedex' | 'battle'>('pokedex');
  pokemonList = signal<Pokemon[]>([]);
  selectedPokemon = signal<Pokemon | null>(null);
  searchTerm = signal('');
  loading = signal(true);
  searchError = signal('');
  currentPage = signal(0);
  pageSize = 30;

  filteredList = computed(() => {
    const term = this.searchTerm().toLowerCase();
    if (!term) return this.pokemonList();
    return this.pokemonList().filter(p =>
      p.name.includes(term) || p.id.toString() === term || p.types.some(t => t.includes(term))
    );
  });

  constructor(private pokemonService: PokemonService) {
    this.loadPokemon();
  }

  async loadPokemon() {
    this.loading.set(true);
    const list = await this.pokemonService.fetchPokemonList(this.pageSize, this.currentPage() * this.pageSize);
    this.pokemonList.set(list);
    this.loading.set(false);
  }

  selectPokemon(pokemon: Pokemon) {
    this.selectedPokemon.set(pokemon);
  }

  closeDetail() {
    this.selectedPokemon.set(null);
  }

  async nextPage() {
    this.currentPage.update(p => p + 1);
    await this.loadPokemon();
  }

  async prevPage() {
    if (this.currentPage() > 0) {
      this.currentPage.update(p => p - 1);
      await this.loadPokemon();
    }
  }

  async onSearch() {
    this.searchError.set('');
    const term = this.searchTerm().trim();
    if (!term) {
      await this.loadPokemon();
      return;
    }
    this.loading.set(true);
    const result = await this.pokemonService.searchPokemon(term);
    if (result) {
      this.pokemonList.set([result]);
      this.searchError.set('');
    } else {
      this.pokemonList.set([]);
      this.searchError.set(`No pokemon found for "${term}"`);
    }
    this.loading.set(false);
  }

  getTypeColor(type: string): string {
    const colors: Record<string, string> = {
      fire: '#F08030', water: '#6890F0', grass: '#78C850', electric: '#F8D030',
      ice: '#98D8D8', fighting: '#C03028', poison: '#A040A0', ground: '#E0C068',
      flying: '#A890F0', psychic: '#F85888', bug: '#A8B820', rock: '#B8A038',
      ghost: '#705898', dragon: '#7038F8', dark: '#705848', steel: '#B8B8D0',
      fairy: '#EE99AC', normal: '#A8A878',
    };
    return colors[type] || '#888';
  }

  getStatBarWidth(value: number): string {
    return Math.min(value / 255 * 100, 100) + '%';
  }

  getStatBarColor(value: number): string {
    if (value >= 100) return '#4caf50';
    if (value >= 60) return '#ff9800';
    return '#f44336';
  }

  formatId(id: number): string {
    return '#' + id.toString().padStart(3, '0');
  }

  capitalize(s: string): string {
    return s.charAt(0).toUpperCase() + s.slice(1);
  }
}
