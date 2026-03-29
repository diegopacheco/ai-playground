import { Component, signal, computed } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { PokemonService, Pokemon } from './pokemon.service';

interface BattleLog {
  text: string;
  type: 'attack' | 'damage' | 'info' | 'victory';
}

@Component({
  selector: 'app-battle',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './battle.component.html',
  styleUrl: './battle.component.css',
})
export class BattleComponent {
  allPokemon = signal<Pokemon[]>([]);
  playerPokemon = signal<Pokemon | null>(null);
  enemyPokemon = signal<Pokemon | null>(null);
  playerHp = signal(0);
  enemyHp = signal(0);
  battleLog = signal<BattleLog[]>([]);
  battleActive = signal(false);
  battleOver = signal(false);
  selecting = signal<'player' | 'enemy' | null>('player');
  searchTerm = signal('');
  loading = signal(true);

  playerHpPercent = computed(() => {
    const p = this.playerPokemon();
    if (!p) return 0;
    return Math.max(0, (this.playerHp() / p.hp) * 100);
  });

  enemyHpPercent = computed(() => {
    const e = this.enemyPokemon();
    if (!e) return 0;
    return Math.max(0, (this.enemyHp() / e.hp) * 100);
  });

  filteredPokemon = computed(() => {
    const term = this.searchTerm().toLowerCase();
    if (!term) return this.allPokemon();
    return this.allPokemon().filter(p =>
      p.name.includes(term) || p.id.toString() === term
    );
  });

  constructor(private pokemonService: PokemonService) {
    this.loadPokemon();
  }

  async loadPokemon() {
    this.loading.set(true);
    const list = await this.pokemonService.fetchPokemonList(50, 0);
    this.allPokemon.set(list);
    this.loading.set(false);
  }

  selectForBattle(pokemon: Pokemon) {
    if (this.selecting() === 'player') {
      this.playerPokemon.set(pokemon);
      this.selecting.set('enemy');
    } else if (this.selecting() === 'enemy') {
      this.enemyPokemon.set(pokemon);
      this.selecting.set(null);
    }
  }

  startBattle() {
    const p = this.playerPokemon();
    const e = this.enemyPokemon();
    if (!p || !e) return;
    this.playerHp.set(p.hp);
    this.enemyHp.set(e.hp);
    this.battleActive.set(true);
    this.battleOver.set(false);
    this.battleLog.set([{
      text: `${this.capitalize(p.name)} vs ${this.capitalize(e.name)} - FIGHT!`,
      type: 'info'
    }]);
  }

  attack() {
    if (this.battleOver()) return;
    const p = this.playerPokemon()!;
    const e = this.enemyPokemon()!;

    const playerDmg = this.calcDamage(p.attack, e.defense);
    this.enemyHp.update(hp => Math.max(0, hp - playerDmg));
    this.battleLog.update(log => [...log,
      { text: `${this.capitalize(p.name)} attacks for ${playerDmg} damage!`, type: 'attack' as const }
    ]);

    if (this.enemyHp() <= 0) {
      this.battleLog.update(log => [...log,
        { text: `${this.capitalize(e.name)} fainted! ${this.capitalize(p.name)} wins!`, type: 'victory' as const }
      ]);
      this.battleOver.set(true);
      return;
    }

    const enemyDmg = this.calcDamage(e.attack, p.defense);
    this.playerHp.update(hp => Math.max(0, hp - enemyDmg));
    this.battleLog.update(log => [...log,
      { text: `${this.capitalize(e.name)} counters for ${enemyDmg} damage!`, type: 'damage' as const }
    ]);

    if (this.playerHp() <= 0) {
      this.battleLog.update(log => [...log,
        { text: `${this.capitalize(p.name)} fainted! ${this.capitalize(e.name)} wins!`, type: 'victory' as const }
      ]);
      this.battleOver.set(true);
    }
  }

  resetBattle() {
    this.playerPokemon.set(null);
    this.enemyPokemon.set(null);
    this.playerHp.set(0);
    this.enemyHp.set(0);
    this.battleLog.set([]);
    this.battleActive.set(false);
    this.battleOver.set(false);
    this.selecting.set('player');
  }

  private calcDamage(attack: number, defense: number): number {
    const base = Math.max(1, Math.floor((attack * 2) / Math.max(1, defense) * 10));
    const variance = 0.85 + Math.random() * 0.3;
    return Math.max(1, Math.floor(base * variance));
  }

  capitalize(s: string): string {
    return s.charAt(0).toUpperCase() + s.slice(1);
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

  hpBarColor(percent: number): string {
    if (percent > 50) return '#4caf50';
    if (percent > 20) return '#ff9800';
    return '#f44336';
  }

  formatId(id: number): string {
    return '#' + id.toString().padStart(3, '0');
  }
}
