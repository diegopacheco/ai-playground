use std::collections::HashMap;
use std::process::Stdio;
use std::time::{Duration, Instant};
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::time::timeout;
use rand::Rng;

const FIRST_NAMES: &[&str] = &[
    "Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Hank",
    "Ivy", "Jack", "Karen", "Leo", "Mona", "Nick", "Olivia", "Pete",
    "Quinn", "Rosa", "Steve", "Tina", "Uma", "Vince", "Wendy", "Xander",
];

const CITIES: &[&str] = &[
    "Tokyo", "Paris", "London", "Berlin", "Sydney", "Toronto", "Mumbai",
    "Cairo", "Seoul", "Lima", "Oslo", "Dublin", "Lisbon", "Prague",
    "Vienna", "Rome", "Athens", "Bangkok", "Hanoi", "Lagos",
    "Nairobi", "Santiago", "Helsinki", "Warsaw",
];

const FOODS: &[&str] = &[
    "sushi", "pizza", "tacos", "ramen", "pasta", "curry", "falafel",
    "paella", "pho", "dumplings", "kebab", "risotto", "burrito",
    "samosa", "pad thai", "goulash", "ceviche", "empanadas",
];

const HOBBIES: &[&str] = &[
    "painting", "chess", "hiking", "photography", "gardening", "cooking",
    "reading", "swimming", "cycling", "yoga", "woodworking", "fishing",
    "archery", "pottery", "dancing", "surfing", "climbing", "sailing",
];

const COLORS: &[&str] = &[
    "red", "blue", "green", "yellow", "purple", "orange", "teal",
    "magenta", "indigo", "crimson", "emerald", "amber", "violet", "coral",
];

struct Fact {
    interaction_id: usize,
    person: String,
    attribute: String,
    value: String,
}

impl Fact {
    fn as_interaction(&self) -> String {
        format!(
            "Interaction #{}: {}'s favorite {} is {}.",
            self.interaction_id, self.person, self.attribute, self.value
        )
    }

    fn question(&self) -> String {
        format!("What is {}'s favorite {}?", self.person, self.attribute)
    }

    fn answer(&self) -> String {
        self.value.clone()
    }
}

fn generate_facts(count: usize) -> Vec<Fact> {
    let mut rng = rand::thread_rng();
    let attributes: Vec<(&str, &[&str])> = vec![
        ("city", CITIES),
        ("food", FOODS),
        ("hobby", HOBBIES),
        ("color", COLORS),
    ];

    (1..=count)
        .map(|i| {
            let person = FIRST_NAMES[rng.gen_range(0..FIRST_NAMES.len())].to_string();
            let (attr, values) = attributes[rng.gen_range(0..attributes.len())];
            let value = values[rng.gen_range(0..values.len())].to_string();
            Fact {
                interaction_id: i,
                person,
                attribute: attr.to_string(),
                value,
            }
        })
        .collect()
}

async fn call_llm(prompt: &str) -> Result<String, String> {
    let mut child = Command::new("claude")
        .args(&["-p", prompt, "--model", "sonnet", "--dangerously-skip-permissions"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to spawn claude: {}", e))?;

    let result = timeout(Duration::from_secs(120), async {
        let mut stdout = child.stdout.take().unwrap();
        let mut output = String::new();
        stdout.read_to_string(&mut output).await.map_err(|e| e.to_string())?;
        child.wait().await.map_err(|e| e.to_string())?;
        Ok::<String, String>(output)
    })
    .await;

    match result {
        Ok(Ok(output)) => {
            let trimmed = output.trim().to_string();
            if trimmed.is_empty() {
                Err("LLM returned empty response".to_string())
            } else {
                Ok(trimmed)
            }
        }
        Ok(Err(e)) => Err(e),
        Err(_) => {
            let _ = child.kill().await;
            Err("LLM timed out after 120s".to_string())
        }
    }
}

fn strategy_raw_context(facts: &[Fact], target: &Fact) -> String {
    let context: String = facts.iter().map(|f| f.as_interaction()).collect::<Vec<_>>().join("\n");
    format!(
        "You have the following interaction history:\n\n{}\n\n\
         Based ONLY on the interactions above, answer this question in ONE word or short phrase.\n\
         Question: {}\nAnswer:",
        context, target.question()
    )
}

fn strategy_summarization(facts: &[Fact], target: &Fact, batch_size: usize) -> String {
    let mut summaries: Vec<String> = Vec::new();
    for chunk in facts.chunks(batch_size) {
        let mut person_attrs: HashMap<String, Vec<String>> = HashMap::new();
        for f in chunk {
            person_attrs
                .entry(f.person.clone())
                .or_default()
                .push(format!("{}: {}", f.attribute, f.value));
        }
        let summary: Vec<String> = person_attrs
            .iter()
            .map(|(person, attrs)| format!("{} -> {}", person, attrs.join(", ")))
            .collect();
        summaries.push(format!(
            "[Summary of interactions #{}-#{}]\n{}",
            chunk.first().unwrap().interaction_id,
            chunk.last().unwrap().interaction_id,
            summary.join("\n")
        ));
    }
    let context = summaries.join("\n\n");
    format!(
        "You have the following summarized interaction history:\n\n{}\n\n\
         Based ONLY on the summaries above, answer this question in ONE word or short phrase.\n\
         Question: {}\nAnswer:",
        context, target.question()
    )
}

fn strategy_rag(facts: &[Fact], target: &Fact, top_k: usize) -> String {
    let query_person = &target.person;
    let query_attr = &target.attribute;

    let mut scored: Vec<(usize, &Fact)> = facts
        .iter()
        .enumerate()
        .map(|(_, f)| {
            let mut score = 0usize;
            if f.person == *query_person {
                score += 10;
            }
            if f.attribute == *query_attr {
                score += 5;
            }
            (score, f)
        })
        .collect();

    scored.sort_by(|a, b| b.0.cmp(&a.0));
    let retrieved: Vec<String> = scored
        .iter()
        .take(top_k)
        .map(|(_, f)| f.as_interaction())
        .collect();

    let context = retrieved.join("\n");
    format!(
        "You retrieved the following relevant interactions:\n\n{}\n\n\
         Based ONLY on the retrieved interactions above, answer this question in ONE word or short phrase.\n\
         Question: {}\nAnswer:",
        context, target.question()
    )
}

fn strategy_knowledge_graph(facts: &[Fact], target: &Fact) -> String {
    let mut graph: HashMap<String, HashMap<String, String>> = HashMap::new();
    for f in facts {
        graph
            .entry(f.person.clone())
            .or_default()
            .insert(f.attribute.clone(), f.value.clone());
    }

    let query_person = &target.person;
    let query_attr = &target.attribute;

    let mut triples: Vec<String> = Vec::new();
    if let Some(attrs) = graph.get(query_person) {
        for (attr, val) in attrs {
            triples.push(format!("({} --[favorite_{}]--> {})", query_person, attr, val));
        }
    }

    let neighbors: Vec<String> = graph
        .iter()
        .filter(|(p, _)| *p != query_person)
        .filter(|(_, attrs)| attrs.contains_key(query_attr))
        .take(5)
        .map(|(p, attrs)| {
            format!("({} --[favorite_{}]--> {})", p, query_attr, attrs[query_attr])
        })
        .collect();

    triples.extend(neighbors);

    let context = if triples.is_empty() {
        "No relevant triples found in knowledge graph.".to_string()
    } else {
        triples.join("\n")
    };

    format!(
        "You have the following knowledge graph triples:\n\n{}\n\n\
         Based ONLY on the triples above, answer this question in ONE word or short phrase.\n\
         Question: {}\nAnswer:",
        context, target.question()
    )
}

fn check_answer(response: &str, expected: &str) -> bool {
    let response_lower = response.to_lowercase();
    let expected_lower = expected.to_lowercase();
    response_lower.contains(&expected_lower)
}

struct BenchResult {
    strategy: String,
    distance: usize,
    correct: bool,
    latency_ms: u128,
    response: String,
    expected: String,
}

async fn run_bench(
    strategy_name: &str,
    prompt: String,
    expected: &str,
    distance: usize,
) -> BenchResult {
    let start = Instant::now();
    let response = match call_llm(&prompt).await {
        Ok(r) => r,
        Err(e) => format!("ERROR: {}", e),
    };
    let latency = start.elapsed().as_millis();
    let correct = check_answer(&response, expected);

    BenchResult {
        strategy: strategy_name.to_string(),
        distance,
        correct,
        latency_ms: latency,
        response: response.chars().take(80).collect(),
        expected: expected.to_string(),
    }
}

#[tokio::main]
async fn main() {
    println!("=== Agent Memory Benchmark ===\n");

    let distances = vec![1, 10, 100, 1000];
    let total_facts = 1000;
    let facts = generate_facts(total_facts);

    let strategies = vec!["raw_context", "summarization", "rag", "knowledge_graph"];

    println!(
        "{:<20} {:<10} {:<10} {:<12} {:<40} {:<20}",
        "Strategy", "Distance", "Correct", "Latency(ms)", "Response", "Expected"
    );
    println!("{}", "-".repeat(112));

    let mut summary: HashMap<String, Vec<bool>> = HashMap::new();

    for &distance in &distances {
        if distance > total_facts {
            continue;
        }
        let target_idx = total_facts - distance;
        let target = &facts[target_idx];
        let facts_up_to_now = &facts[..total_facts];

        for &strategy in &strategies {
            let prompt = match strategy {
                "raw_context" => {
                    if distance > 100 {
                        let window_start = if target_idx > 50 { target_idx - 50 } else { 0 };
                        let window_end = std::cmp::min(target_idx + 50, total_facts);
                        let windowed: Vec<Fact> = facts_up_to_now[window_start..window_end]
                            .iter()
                            .map(|f| Fact {
                                interaction_id: f.interaction_id,
                                person: f.person.clone(),
                                attribute: f.attribute.clone(),
                                value: f.value.clone(),
                            })
                            .collect();
                        strategy_raw_context(&windowed, target)
                    } else {
                        strategy_raw_context(facts_up_to_now, target)
                    }
                }
                "summarization" => strategy_summarization(facts_up_to_now, target, 50),
                "rag" => strategy_rag(facts_up_to_now, target, 10),
                "knowledge_graph" => strategy_knowledge_graph(facts_up_to_now, target),
                _ => unreachable!(),
            };

            let result = run_bench(strategy, prompt, &target.answer(), distance).await;

            println!(
                "{:<20} {:<10} {:<10} {:<12} {:<40} {:<20}",
                result.strategy,
                result.distance,
                if result.correct { "YES" } else { "NO" },
                result.latency_ms,
                result.response,
                result.expected
            );

            summary
                .entry(result.strategy.clone())
                .or_default()
                .push(result.correct);
        }
    }

    println!("\n=== Summary ===\n");
    println!("{:<20} {:<10} {:<10} {:<10}", "Strategy", "Correct", "Total", "Accuracy");
    println!("{}", "-".repeat(50));

    for strategy in &strategies {
        if let Some(results) = summary.get(*strategy) {
            let correct = results.iter().filter(|&&c| c).count();
            let total = results.len();
            let accuracy = (correct as f64 / total as f64) * 100.0;
            println!(
                "{:<20} {:<10} {:<10} {:.1}%",
                strategy, correct, total, accuracy
            );
        }
    }

    println!("\n=== Benchmark Complete ===");
}
