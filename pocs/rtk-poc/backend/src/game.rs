use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::Serialize;

pub const SYMBOLS: [&str; 8] = ["🍎", "🍌", "🍒", "🍇", "🍋", "🍓", "🍉", "🥝"];

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct Card {
    pub id: u32,
    pub symbol: String,
}

pub fn build_deck() -> Vec<Card> {
    let mut deck: Vec<Card> = SYMBOLS
        .iter()
        .enumerate()
        .flat_map(|(i, s)| {
            vec![
                Card { id: (i as u32) * 2, symbol: s.to_string() },
                Card { id: (i as u32) * 2 + 1, symbol: s.to_string() },
            ]
        })
        .collect();
    deck.shuffle(&mut thread_rng());
    deck
}

pub fn pair_count(deck: &[Card]) -> usize {
    deck.len() / 2
}
