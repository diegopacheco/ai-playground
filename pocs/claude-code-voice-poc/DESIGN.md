# Claude Code Voice POC - Design Doc

Build with claude code using /voice commands. This is a POC to test the voice feature and how it can be used to build an app. The app will be a Pokedex-like card battle game using TanStack libraries, React 19, and Bun. The design doc will be updated with all prompts and decisions made during the implementation.

## Prompts

1. "Don't write any code yet. We will write a design doc and after that, then we implement. But for now, I'm not gonna implement anything."
2. "So all the prompts that they're gonna write here they might be recorded on the design doc. So make sure from now on that all prompts, they go to the design doc."
3. "I'm testing the voice feature. Also, you need to have a section like experience notes, but this will be on my README."
4. "Build an application using TanStack (Tanner Linsley's libraries), use as many TanStack modules as possible. React 19, Bun. Build a Pokedex-like card battle game: shuffle cards, two players open cards and battle, after three rounds declare the champion. Cool animations. Have a run.sh. No Redux."
5. "The Pokedex game needs tabs: Tab 1 - each player selects and opens their cards. Tab 2 - show all Pokemons by category. Tab 3 - match/fight. Tab 4 - history of all fights, click to see score. Best of 3 rounds. Cool animations."
6. "Update design doc, make README cool. Once done implementing, run the app in the browser, use Playwright MCP to take screenshots of each tab, and update README with the screenshots."
7. "I want unit tests, integration tests, and Playwright tests. Have a test.sh script that runs all tests. Always keep design doc and README in sync."

## Application Design

### Tech Stack
* React 19
* Bun
* TanStack Router (routing/tabs)
* TanStack Query (data fetching - PokeAPI)
* TanStack Table (battle history display)
* TanStack Form (player name input)
* TanStack Virtual (Pokedex scrolling)
* CSS animations (card flip, battle clash, champion entrance)
* No Redux

### Features
* Pokedex battle card game
* Two players
* 5 Tabs:
  * Home: Enter player names (TanStack Form)
  * Cards: Each player selects and opens/flips their cards
  * Pokedex: Browse all 151 Gen 1 Pokemon by type (TanStack Virtual)
  * Battle: The fight happens here with animations, 3 rounds
  * History: List of all past fights with TanStack Table, click to see detailed score
* Cards are shuffled and dealt face-down
* Each round both players flip a card revealing a Pokemon
* Pokemon stats (HP + ATK + DEF + SPD = Power) determine the winner
* Best of 3 rounds determines the champion
* Cool card flip, battle clash, and champion entrance animations
* Fetches real Pokemon data from PokeAPI

### Testing
* Unit tests: store logic (bun test)
* E2E tests: Playwright (navigation, form, tabs, filtering)
* `./test.sh` - runs all tests

### How to Run
* `./run.sh` - starts the app with Bun
