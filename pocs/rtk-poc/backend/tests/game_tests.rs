use memory_backend::game::{Card, SYMBOLS, build_deck, pair_count};
use std::collections::HashSet;

#[test]
fn deck_has_16_cards() {
    assert_eq!(build_deck().len(), 16);
}

#[test]
fn deck_has_8_pairs() {
    assert_eq!(pair_count(&build_deck()), 8);
}

#[test]
fn deck_each_symbol_appears_twice() {
    let deck = build_deck();
    for symbol in SYMBOLS.iter() {
        let count = deck.iter().filter(|c| c.symbol == *symbol).count();
        assert_eq!(count, 2, "symbol {symbol} should appear twice");
    }
}

#[test]
fn deck_has_unique_ids() {
    let deck = build_deck();
    let ids: HashSet<u32> = deck.iter().map(|c| c.id).collect();
    assert_eq!(ids.len(), 16);
}

#[test]
fn deck_ids_in_range_zero_to_fifteen() {
    let deck = build_deck();
    for card in &deck {
        assert!(card.id < 16, "id {} out of range", card.id);
    }
}

#[test]
fn deck_is_likely_shuffled() {
    let canonical: Vec<Card> = SYMBOLS
        .iter()
        .enumerate()
        .flat_map(|(i, s)| {
            vec![
                Card { id: (i as u32) * 2, symbol: s.to_string() },
                Card { id: (i as u32) * 2 + 1, symbol: s.to_string() },
            ]
        })
        .collect();
    let mut differ = 0;
    for _ in 0..5 {
        if build_deck() != canonical {
            differ += 1;
        }
    }
    assert!(differ >= 4, "deck should be shuffled in at least 4/5 runs");
}

#[test]
fn pair_count_handles_empty() {
    let empty: Vec<Card> = Vec::new();
    assert_eq!(pair_count(&empty), 0);
}

#[test]
fn symbols_constant_has_eight_entries() {
    assert_eq!(SYMBOLS.len(), 8);
}
