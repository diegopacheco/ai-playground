use std::io::{self, BufRead, Read, Write};
use std::net::TcpStream;

pub struct Request {
    pub method: String,
    pub path: String,
    pub body: String,
}

pub struct Response {
    pub status: u16,
    pub body: String,
}

impl Response {
    pub fn json(status: u16, body: String) -> Self {
        Self { status, body }
    }
}

pub fn read_request(reader: &mut impl BufRead) -> io::Result<Request> {
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or_default().to_string();
    let target = parts.next().unwrap_or_default();
    let path = target.split('?').next().unwrap_or_default().to_string();
    let mut length = 0;
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header)? == 0 {
            break;
        }
        let header = header.trim_end();
        if header.is_empty() {
            break;
        }
        if let Some((name, value)) = header.split_once(':') {
            if name.trim().eq_ignore_ascii_case("content-length") {
                length = value.trim().parse().unwrap_or(0);
            }
        }
    }
    let mut body = vec![0; length];
    reader.read_exact(&mut body)?;
    Ok(Request { method, path, body: String::from_utf8_lossy(&body).into_owned() })
}

pub fn write_response(writer: &mut impl Write, response: &Response) -> io::Result<()> {
    let head = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        response.status,
        reason(response.status),
        response.body.len()
    );
    writer.write_all(head.as_bytes())?;
    writer.write_all(response.body.as_bytes())?;
    writer.flush()
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        502 => "Bad Gateway",
        _ => "Error",
    }
}

pub fn post_json(addr: &str, path: &str, body: &str) -> Result<String, String> {
    let mut stream = TcpStream::connect(addr).map_err(|error| format!("Unable to reach {addr}: {error}"))?;
    let request = format!("POST {path} HTTP/1.0\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len());
    stream.write_all(request.as_bytes()).map_err(|error| error.to_string())?;
    let mut raw = String::new();
    stream.read_to_string(&mut raw).map_err(|error| error.to_string())?;
    let (head, payload) = raw.split_once("\r\n\r\n").ok_or_else(|| format!("Malformed response from {addr}"))?;
    let status = head.split_whitespace().nth(1).unwrap_or_default();
    if status != "200" {
        return Err(format!("{addr}{path} returned {status}: {payload}"));
    }
    Ok(payload.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn the_body_is_read_by_content_length_so_the_connection_is_not_drained_forever() {
        let raw = "POST /api/ask?x=1 HTTP/1.1\r\nHost: a\r\ncontent-length: 5\r\n\r\nhello and more";
        let request = read_request(&mut Cursor::new(raw)).unwrap();
        assert_eq!(request.method, "POST");
        assert_eq!(request.path, "/api/ask");
        assert_eq!(request.body, "hello");
    }

    #[test]
    fn content_length_counts_bytes_not_characters() {
        let mut out = Vec::new();
        write_response(&mut out, &Response::json(200, "\"é\"".into())).unwrap();
        assert!(String::from_utf8(out).unwrap().contains("Content-Length: 4\r\n"));
    }
}
