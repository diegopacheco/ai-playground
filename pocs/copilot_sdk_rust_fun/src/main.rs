use copilot_sdk::{Client, SessionConfig, Tool, ToolHandler, ToolResult};
use reqwest::Url;
use std::io::{self, Write};
use std::sync::Arc;

fn other_error(msg: impl Into<String>) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::Other, msg.into())
}

fn weather_description(code: i64) -> &'static str {
    match code {
        0 => "Clear",
        1 | 2 | 3 => "Partly cloudy",
        45 | 48 => "Fog",
        51 | 53 | 55 => "Drizzle",
        56 | 57 => "Freezing drizzle",
        61 | 63 | 65 => "Rain",
        66 | 67 => "Freezing rain",
        71 | 73 | 75 => "Snow",
        77 => "Snow grains",
        80 | 81 | 82 => "Rain showers",
        85 | 86 => "Snow showers",
        95 => "Thunderstorm",
        96 | 99 => "Thunderstorm with hail",
        _ => "Unknown",
    }
}

async fn fetch_weather(city: &str) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let mut geo_url = Url::parse("https://geocoding-api.open-meteo.com/v1/search")?;
    geo_url
        .query_pairs_mut()
        .append_pair("name", city)
        .append_pair("count", "1")
        .append_pair("language", "en")
        .append_pair("format", "json");

    let geo: serde_json::Value = reqwest::get(geo_url).await?.json().await?;
    let results = geo
        .get("results")
        .and_then(|v| v.as_array())
        .ok_or_else(|| other_error("City not found"))?;
    let first = results
        .get(0)
        .ok_or_else(|| other_error("City not found"))?;

    let latitude = first
        .get("latitude")
        .and_then(|v| v.as_f64())
        .ok_or_else(|| other_error("Missing latitude"))?;
    let longitude = first
        .get("longitude")
        .and_then(|v| v.as_f64())
        .ok_or_else(|| other_error("Missing longitude"))?;
    let name = first
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| other_error("Missing name"))?;
    let country = first
        .get("country")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let mut forecast_url = Url::parse("https://api.open-meteo.com/v1/forecast")?;
    forecast_url
        .query_pairs_mut()
        .append_pair("latitude", &latitude.to_string())
        .append_pair("longitude", &longitude.to_string())
        .append_pair("current", "temperature_2m,wind_speed_10m,weather_code")
        .append_pair("current_weather", "true")
        .append_pair("temperature_unit", "celsius")
        .append_pair("wind_speed_unit", "ms");

    let forecast: serde_json::Value = reqwest::get(forecast_url).await?.json().await?;
    if forecast.get("error").and_then(|v| v.as_bool()).unwrap_or(false) {
        let reason = forecast
            .get("reason")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown error");
        return Err(other_error(reason).into());
    }

    let current = forecast.get("current").and_then(|v| v.as_object());
    let current_weather = forecast.get("current_weather").and_then(|v| v.as_object());
    let current = current.or(current_weather).ok_or_else(|| other_error("Missing current data"))?;

    let temperature = current
        .get("temperature_2m")
        .and_then(|v| v.as_f64())
        .or_else(|| current.get("temperature").and_then(|v| v.as_f64()))
        .ok_or_else(|| other_error("Missing temperature"))?;
    let wind = current
        .get("wind_speed_10m")
        .and_then(|v| v.as_f64())
        .or_else(|| current.get("windspeed").and_then(|v| v.as_f64()))
        .ok_or_else(|| other_error("Missing wind"))?;
    let code = current
        .get("weather_code")
        .and_then(|v| v.as_i64())
        .or_else(|| current.get("weathercode").and_then(|v| v.as_i64()))
        .ok_or_else(|| other_error("Missing weather code"))?;

    let description = weather_description(code);
    let location = if country.is_empty() {
        name.to_string()
    } else {
        format!("{}, {}", name, country)
    };

    Ok(format!(
        "{}: {} C, wind {} m/s, {}",
        location, temperature, wind, description
    ))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let client = Client::builder().build()?;
    client.start().await?;

    let tool = Tool::new("city_weather")
        .description("Get current weather")
        .schema(serde_json::json!({
            "type": "object",
            "properties": {
                "city": { "type": "string", "description": "City name" }
            },
            "required": ["city"]
        }));

    let config = SessionConfig {
        tools: vec![tool.clone()],
        ..Default::default()
    };
    let session = client.create_session(config).await?;

    let handler: ToolHandler = Arc::new(move |_name, args| {
        let city = args
            .get("city")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        if city.is_empty() {
            return ToolResult::error("City is required");
        }
        let city_clone = city.clone();
        let result = std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|e| e.to_string())?;
            rt.block_on(fetch_weather(&city_clone))
                .map_err(|e| e.to_string())
        })
        .join()
        .unwrap_or_else(|_| Err("Failed to run weather task".to_string()))
        .unwrap_or_else(|e| e);
        ToolResult::text(result)
    });

    session
        .register_tool_with_handler(tool, Some(handler))
        .await;

    loop {
        print!("city> ");
        io::stdout().flush()?;

        let mut line = String::new();
        if io::stdin().read_line(&mut line)? == 0 {
            break;
        }
        let city = line.trim().to_string();
        if city.is_empty() {
            continue;
        }
        if city.eq_ignore_ascii_case("exit") || city.eq_ignore_ascii_case("quit") {
            break;
        }

        let args = serde_json::json!({ "city": city });
        let result = session.invoke_tool("city_weather", &args).await?;
        println!("{}", result.text_result_for_llm);
    }

    client.stop().await?;
    Ok(())
}
