use scraper::{Html, Selector, ElementRef};

pub async fn web_search(url: &str) -> String {
    if url.is_empty() {
        return "Error: URL cannot be empty".to_string();
    }

    let client = reqwest::Client::new();
    let response = match client.get(url).send().await {
        Ok(resp) => resp,
        Err(e) => return format!("Error fetching URL: {}", e),
    };

    let html_content = match response.text().await {
        Ok(text) => text,
        Err(e) => return format!("Error reading response: {}", e),
    };

    extract_text_from_html(&html_content)
}

fn is_excluded_tag(element: &ElementRef) -> bool {
    let tag_name = element.value().name();
    tag_name == "script" || tag_name == "style"
}

fn extract_text_recursive(element: ElementRef, text_content: &mut String) {
    if is_excluded_tag(&element) {
        return;
    }

    for child in element.children() {
        if let Some(child_element) = ElementRef::wrap(child) {
            extract_text_recursive(child_element, text_content);
        } else if let Some(text) = child.value().as_text() {
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                if !text_content.is_empty() {
                    text_content.push(' ');
                }
                text_content.push_str(trimmed);
            }
        }
    }
}

pub fn extract_text_from_html(html: &str) -> String {
    let document = Html::parse_document(html);
    let body_selector = Selector::parse("body").unwrap();

    let mut text_content = String::new();

    if let Some(body) = document.select(&body_selector).next() {
        extract_text_recursive(body, &mut text_content);
    } else {
        for child in document.root_element().children() {
            if let Some(element) = ElementRef::wrap(child) {
                extract_text_recursive(element, &mut text_content);
            } else if let Some(text) = child.value().as_text() {
                let trimmed = text.trim();
                if !trimmed.is_empty() {
                    if !text_content.is_empty() {
                        text_content.push(' ');
                    }
                    text_content.push_str(trimmed);
                }
            }
        }
    }

    text_content
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_text_simple_html() {
        let html = "<html><body><p>Hello World</p></body></html>";
        let result = extract_text_from_html(html);
        assert_eq!(result, "Hello World");
    }

    #[test]
    fn test_extract_text_strips_script() {
        let html = "<html><body><script>var x = 1;</script><p>Content</p></body></html>";
        let result = extract_text_from_html(html);
        assert!(!result.contains("var x"));
        assert!(result.contains("Content"));
    }

    #[test]
    fn test_extract_text_strips_style() {
        let html = "<html><body><style>.class { color: red; }</style><p>Text</p></body></html>";
        let result = extract_text_from_html(html);
        assert!(!result.contains("color"));
        assert!(result.contains("Text"));
    }

    #[test]
    fn test_extract_text_multiple_elements() {
        let html = "<html><body><h1>Title</h1><p>Paragraph</p><div>Div content</div></body></html>";
        let result = extract_text_from_html(html);
        assert!(result.contains("Title"));
        assert!(result.contains("Paragraph"));
        assert!(result.contains("Div content"));
    }

    #[test]
    fn test_extract_text_nested_elements() {
        let html = "<html><body><div><span><a>Link text</a></span></div></body></html>";
        let result = extract_text_from_html(html);
        assert!(result.contains("Link text"));
    }

    #[test]
    fn test_extract_text_empty_html() {
        let html = "";
        let result = extract_text_from_html(html);
        assert!(result.is_empty());
    }

    #[test]
    fn test_extract_text_only_whitespace() {
        let html = "<html><body>   \n\t   </body></html>";
        let result = extract_text_from_html(html);
        assert!(result.is_empty());
    }

    #[test]
    fn test_extract_text_preserves_content_order() {
        let html = "<html><body><p>First</p><p>Second</p><p>Third</p></body></html>";
        let result = extract_text_from_html(html);
        let first_pos = result.find("First").unwrap();
        let second_pos = result.find("Second").unwrap();
        let third_pos = result.find("Third").unwrap();
        assert!(first_pos < second_pos);
        assert!(second_pos < third_pos);
    }

    #[test]
    fn test_extract_text_complex_page() {
        let html = r#"
        <html>
        <head>
            <title>Test Page</title>
            <style>body { margin: 0; }</style>
            <script>console.log('test');</script>
        </head>
        <body>
            <header>
                <nav>Navigation</nav>
            </header>
            <main>
                <article>
                    <h1>Main Article</h1>
                    <p>This is the main content.</p>
                </article>
            </main>
            <footer>Footer text</footer>
            <script>document.ready();</script>
        </body>
        </html>
        "#;
        let result = extract_text_from_html(html);
        assert!(result.contains("Navigation"));
        assert!(result.contains("Main Article"));
        assert!(result.contains("main content"));
        assert!(result.contains("Footer text"));
        assert!(!result.contains("console.log"));
        assert!(!result.contains("document.ready"));
        assert!(!result.contains("margin"));
    }

    #[tokio::test]
    async fn test_web_search_empty_url() {
        let result = web_search("").await;
        assert_eq!(result, "Error: URL cannot be empty");
    }

    #[tokio::test]
    async fn test_web_search_invalid_url() {
        let result = web_search("not-a-valid-url").await;
        assert!(result.starts_with("Error"));
    }
}
