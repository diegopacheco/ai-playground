import { Injectable } from '@angular/core';

export interface Pokemon {
  id: number;
  name: string;
  types: string[];
  sprite: string;
  hp: number;
  attack: number;
  defense: number;
  speed: number;
  height: number;
  weight: number;
}

@Injectable({ providedIn: 'root' })
export class PokemonService {
  private baseUrl = 'https://pokeapi.co/api/v2/pokemon';

  async fetchPokemonList(limit: number = 30, offset: number = 0): Promise<Pokemon[]> {
    const res = await fetch(`${this.baseUrl}?limit=${limit}&offset=${offset}`);
    const data = await res.json();
    const promises = data.results.map((p: { url: string }) => this.fetchPokemonByUrl(p.url));
    return Promise.all(promises);
  }

  async fetchPokemonById(id: number): Promise<Pokemon> {
    return this.fetchPokemonByUrl(`${this.baseUrl}/${id}`);
  }

  async searchPokemon(name: string): Promise<Pokemon | null> {
    try {
      return await this.fetchPokemonByUrl(`${this.baseUrl}/${name.toLowerCase()}`);
    } catch {
      return null;
    }
  }

  private async fetchPokemonByUrl(url: string): Promise<Pokemon> {
    const res = await fetch(url);
    const data = await res.json();
    return {
      id: data.id,
      name: data.name,
      types: data.types.map((t: { type: { name: string } }) => t.type.name),
      sprite: data.sprites.other['official-artwork'].front_default || data.sprites.front_default,
      hp: data.stats[0].base_stat,
      attack: data.stats[1].base_stat,
      defense: data.stats[2].base_stat,
      speed: data.stats[5].base_stat,
      height: data.height,
      weight: data.weight,
    };
  }
}
