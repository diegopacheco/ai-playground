use regex::Regex;
use reqwest::blocking::Client;
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::Duration;

#[derive(Debug, Clone)]
struct Paper {
    id: String,
    title: String,
    pdf_url: String,
}

#[derive(Serialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<Message>,
    max_completion_tokens: u32,
}

#[derive(Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct OpenAIResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: ResponseMessage,
}

#[derive(Deserialize)]
struct ResponseMessage {
    content: String,
}

fn main() {
    let papers_dir = Path::new("papers");
    let summary_dir = Path::new("summary");

    fs::create_dir_all(papers_dir).expect("Failed to create papers directory");
    fs::create_dir_all(summary_dir).expect("Failed to create summary directory");

    let existing_summaries = get_existing_summaries(summary_dir);
    println!("Found {} existing summaries", existing_summaries.len());

    let client = Client::builder()
        .timeout(Duration::from_secs(120))
        .user_agent("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36")
        .build()
        .expect("Failed to create HTTP client");

    println!("Fetching papers from arXiv...");
    let papers = fetch_arxiv_papers(&client);
    println!("Found {} papers", papers.len());

    let papers_to_process: Vec<Paper> = papers
        .into_iter()
        .filter(|p| !existing_summaries.contains(&sanitize_filename(&p.title)))
        .collect();

    println!("{} papers need processing", papers_to_process.len());

    let openai_key = std::env::var("OPEN_AI_API_KEY").expect("OPEN_AI_API_KEY environment variable not set");

    for (i, paper) in papers_to_process.iter().enumerate() {
        println!("\n[{}/{}] Processing: {}", i + 1, papers_to_process.len(), paper.title);

        let pdf_filename = format!("{}.pdf", sanitize_filename(&paper.title));
        let pdf_path = papers_dir.join(&pdf_filename);

        if !pdf_path.exists() {
            println!("  Downloading PDF...");
            match download_pdf(&client, &paper.pdf_url, &pdf_path) {
                Ok(_) => println!("  PDF saved: {}", pdf_filename),
                Err(e) => {
                    println!("  Failed to download PDF: {}", e);
                    continue;
                }
            }
        } else {
            println!("  PDF already exists");
        }

        println!("  Generating summary...");
        match generate_summary(&client, &openai_key, paper) {
            Ok(summary) => {
                let summary_filename = format!("{}-summary.md", sanitize_filename(&paper.title));
                let summary_path = summary_dir.join(&summary_filename);
                fs::write(&summary_path, summary).expect("Failed to write summary");
                println!("  Summary saved: {}", summary_filename);
            }
            Err(e) => {
                println!("  Failed to generate summary: {}", e);
            }
        }

        std::thread::sleep(Duration::from_millis(500));
    }

    println!("\nDone!");
}

fn get_existing_summaries(summary_dir: &Path) -> HashSet<String> {
    let mut summaries = HashSet::new();
    if let Ok(entries) = fs::read_dir(summary_dir) {
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                if name.ends_with("-summary.md") {
                    let paper_name = name.trim_end_matches("-summary.md").to_string();
                    summaries.insert(paper_name);
                }
            }
        }
    }
    summaries
}

fn sanitize_filename(name: &str) -> String {
    let re = Regex::new(r#"[<>:"/\\|?*\x00-\x1f]"#).unwrap();
    let sanitized = re.replace_all(name, "_").to_string();
    let sanitized = sanitized.trim().to_string();
    if sanitized.len() > 200 {
        sanitized[..200].to_string()
    } else {
        sanitized
    }
}

fn fetch_arxiv_papers(client: &Client) -> Vec<Paper> {
    let mut all_papers = Vec::new();
    let base_url = "https://arxiv.org/list/cs.AI/recent";

    let response = client.get(base_url).send().expect("Failed to fetch arXiv page");
    let html = response.text().expect("Failed to read response");
    let document = Html::parse_document(&html);

    let dt_selector = Selector::parse("dt").unwrap();
    let dd_selector = Selector::parse("dd").unwrap();
    let a_selector = Selector::parse("a").unwrap();
    let title_selector = Selector::parse("div.list-title").unwrap();

    let dts: Vec<_> = document.select(&dt_selector).collect();
    let dds: Vec<_> = document.select(&dd_selector).collect();

    let id_regex = Regex::new(r"/abs/(\d+\.\d+)").unwrap();

    for (dt, dd) in dts.iter().zip(dds.iter()) {
        if all_papers.len() >= 100 {
            break;
        }

        let mut paper_id = String::new();
        for a in dt.select(&a_selector) {
            if let Some(href) = a.value().attr("href") {
                if let Some(caps) = id_regex.captures(href) {
                    paper_id = caps.get(1).unwrap().as_str().to_string();
                    break;
                }
            }
        }

        if paper_id.is_empty() {
            continue;
        }

        let mut title = String::new();
        for div in dd.select(&title_selector) {
            title = div.text().collect::<String>();
            title = title.replace("Title:", "").trim().to_string();
            break;
        }

        if title.is_empty() {
            title = format!("Paper-{}", paper_id);
        }

        all_papers.push(Paper {
            id: paper_id.clone(),
            title,
            pdf_url: format!("https://arxiv.org/pdf/{}.pdf", paper_id),
        });
    }

    if all_papers.len() < 100 {
        let show_url = "https://arxiv.org/list/cs.AI/recent?skip=0&show=100";
        if let Ok(response) = client.get(show_url).send() {
            if let Ok(html) = response.text() {
                let document = Html::parse_document(&html);
                let dts: Vec<_> = document.select(&dt_selector).collect();
                let dds: Vec<_> = document.select(&dd_selector).collect();

                for (dt, dd) in dts.iter().zip(dds.iter()) {
                    if all_papers.len() >= 100 {
                        break;
                    }

                    let mut paper_id = String::new();
                    for a in dt.select(&a_selector) {
                        if let Some(href) = a.value().attr("href") {
                            if let Some(caps) = id_regex.captures(href) {
                                paper_id = caps.get(1).unwrap().as_str().to_string();
                                break;
                            }
                        }
                    }

                    if paper_id.is_empty() {
                        continue;
                    }

                    if all_papers.iter().any(|p| p.id == paper_id) {
                        continue;
                    }

                    let mut title = String::new();
                    for div in dd.select(&title_selector) {
                        title = div.text().collect::<String>();
                        title = title.replace("Title:", "").trim().to_string();
                        break;
                    }

                    if title.is_empty() {
                        title = format!("Paper-{}", paper_id);
                    }

                    all_papers.push(Paper {
                        id: paper_id.clone(),
                        title,
                        pdf_url: format!("https://arxiv.org/pdf/{}.pdf", paper_id),
                    });
                }
            }
        }
    }

    all_papers
}

fn download_pdf(client: &Client, url: &str, path: &Path) -> Result<(), String> {
    let response = client.get(url).send().map_err(|e| e.to_string())?;
    let bytes = response.bytes().map_err(|e| e.to_string())?;
    let mut file = fs::File::create(path).map_err(|e| e.to_string())?;
    file.write_all(&bytes).map_err(|e| e.to_string())?;
    Ok(())
}

fn generate_summary(client: &Client, api_key: &str, paper: &Paper) -> Result<String, String> {
    let prompt = format!(
        "Please provide a comprehensive summary of the following academic paper from arXiv:\n\n\
        Title: {}\n\
        arXiv ID: {}\n\
        PDF URL: {}\n\n\
        Please structure your summary with the following sections:\n\
        1. **Overview**: Brief description of what the paper is about\n\
        2. **Key Contributions**: Main contributions and innovations\n\
        3. **Methodology**: Approach and methods used\n\
        4. **Results**: Key findings and results\n\
        5. **Implications**: Potential impact and applications\n\n\
        Note: Since I cannot access the full PDF content, please provide a summary based on the title \
        and your knowledge of similar research in this area. If you have knowledge about this specific paper, \
        please use it.",
        paper.title, paper.id, paper.pdf_url
    );

    let request = OpenAIRequest {
        model: "gpt-4o-mini".to_string(),
        messages: vec![Message {
            role: "user".to_string(),
            content: prompt,
        }],
        max_completion_tokens: 2000,
    };

    let response = client
        .post("https://api.openai.com/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .map_err(|e| e.to_string())?;

    let status = response.status();
    let body = response.text().map_err(|e| e.to_string())?;

    if !status.is_success() {
        return Err(format!("API error {}: {}", status, body));
    }

    let api_response: OpenAIResponse = serde_json::from_str(&body).map_err(|e| format!("Parse error: {} - Body: {}", e, body))?;

    if api_response.choices.is_empty() {
        return Err("No response from API".to_string());
    }

    let summary_content = &api_response.choices[0].message.content;

    let full_summary = format!(
        "# {}\n\n**arXiv ID**: {}\n**PDF**: {}\n\n---\n\n{}",
        paper.title, paper.id, paper.pdf_url, summary_content
    );

    Ok(full_summary)
}
