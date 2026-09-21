use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;

#[derive(Debug, Clone, PartialEq)]
pub enum Reply {
    Status(String),
    Error(String),
    Integer(i64),
    Bulk(Option<Vec<u8>>),
    Array(Vec<Reply>),
}

impl Reply {
    pub fn text(&self) -> Option<String> {
        match self {
            Reply::Status(value) => Some(value.clone()),
            Reply::Bulk(Some(bytes)) => Some(String::from_utf8_lossy(bytes).into_owned()),
            Reply::Integer(value) => Some(value.to_string()),
            _ => None,
        }
    }

    pub fn integer(&self) -> Option<i64> {
        match self {
            Reply::Integer(value) => Some(*value),
            other => other.text()?.parse().ok(),
        }
    }

    pub fn items(&self) -> &[Reply] {
        match self {
            Reply::Array(items) => items,
            _ => &[],
        }
    }
}

pub fn encode(args: &[&[u8]]) -> Vec<u8> {
    let mut out = format!("*{}\r\n", args.len()).into_bytes();
    for arg in args {
        out.extend_from_slice(format!("${}\r\n", arg.len()).as_bytes());
        out.extend_from_slice(arg);
        out.extend_from_slice(b"\r\n");
    }
    out
}

pub fn decode(reader: &mut impl BufRead) -> Result<Reply, String> {
    let mut line = String::new();
    reader.read_line(&mut line).map_err(|error| error.to_string())?;
    let line = line.trim_end_matches("\r\n");
    if line.is_empty() {
        return Err("redis closed the connection".into());
    }
    let (kind, rest) = line.split_at(1);
    let number = || rest.parse::<i64>().map_err(|error| format!("bad redis length {rest}: {error}"));
    match kind {
        "+" => Ok(Reply::Status(rest.into())),
        "-" => Ok(Reply::Error(rest.into())),
        ":" => Ok(Reply::Integer(number()?)),
        "$" => {
            let length = number()?;
            if length < 0 {
                return Ok(Reply::Bulk(None));
            }
            let mut bytes = vec![0; length as usize + 2];
            reader.read_exact(&mut bytes).map_err(|error| error.to_string())?;
            bytes.truncate(length as usize);
            Ok(Reply::Bulk(Some(bytes)))
        }
        "*" => {
            let length = number()?;
            (0..length.max(0)).map(|_| decode(reader)).collect::<Result<Vec<_>, _>>().map(Reply::Array)
        }
        other => Err(format!("unknown redis reply type {other}")),
    }
}

pub struct Redis {
    addr: String,
}

impl Redis {
    pub fn new(addr: &str) -> Self {
        Self { addr: addr.to_string() }
    }

    pub fn command(&self, args: &[&[u8]]) -> Result<Reply, String> {
        let mut stream = TcpStream::connect(&self.addr).map_err(|error| format!("Unable to reach redis at {}: {error}", self.addr))?;
        stream.write_all(&encode(args)).map_err(|error| error.to_string())?;
        match decode(&mut BufReader::new(stream))? {
            Reply::Error(message) => Err(message),
            reply => Ok(reply),
        }
    }

    pub fn words(&self, args: &[&str]) -> Result<Reply, String> {
        let bytes: Vec<&[u8]> = args.iter().map(|arg| arg.as_bytes()).collect();
        self.command(&bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn binary_vectors_are_length_prefixed_so_crlf_bytes_inside_them_are_safe() {
        let vector: &[u8] = &[b'\r', b'\n', 0];
        assert_eq!(encode(&[b"SET", vector]), b"*2\r\n$3\r\nSET\r\n$3\r\n\r\n\0\r\n".to_vec());
    }

    #[test]
    fn a_search_reply_decodes_into_nested_documents() {
        let raw = "*3\r\n:1\r\n$4\r\nqa:1\r\n*2\r\n$8\r\ndistance\r\n$4\r\n0.05\r\n";
        let reply = decode(&mut Cursor::new(raw)).unwrap();
        assert_eq!(reply.items()[0].integer(), Some(1));
        assert_eq!(reply.items()[2].items()[1].text(), Some("0.05".into()));
    }

    #[test]
    fn missing_keys_and_errors_are_distinct_from_empty_values() {
        assert_eq!(decode(&mut Cursor::new("$-1\r\n")).unwrap(), Reply::Bulk(None));
        assert_eq!(decode(&mut Cursor::new("-ERR boom\r\n")).unwrap(), Reply::Error("ERR boom".into()));
    }
}
