use std::collections::BTreeMap;
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::Command;

const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const RED: &str = "\x1b[31m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const MAGENTA: &str = "\x1b[35m";
const CYAN: &str = "\x1b[36m";
const WHITE: &str = "\x1b[37m";

struct Entry {
    port: u16,
    pid: u32,
    name: String,
    path: String,
}

fn parse_port(name: &str) -> Option<u16> {
    let idx = name.rfind(':')?;
    name[idx + 1..].parse::<u16>().ok()
}

fn exe_path(pid: u32, cache: &mut BTreeMap<u32, String>) -> String {
    if let Some(p) = cache.get(&pid) {
        return p.clone();
    }
    let out = Command::new("/bin/ps")
        .args(["-p", &pid.to_string(), "-o", "comm="])
        .output();
    let path = match out {
        Ok(o) => String::from_utf8_lossy(&o.stdout).trim().to_string(),
        Err(_) => String::new(),
    };
    cache.insert(pid, path.clone());
    path
}

fn basename(path: &str) -> String {
    path.rsplit('/').next().unwrap_or(path).to_string()
}

fn scan() -> Vec<Entry> {
    let out = Command::new("/usr/sbin/lsof")
        .args(["-nP", "-iTCP", "-sTCP:LISTEN", "+c", "0", "-F", "pcn"])
        .output();
    let text = match out {
        Ok(o) => String::from_utf8_lossy(&o.stdout).into_owned(),
        Err(_) => return Vec::new(),
    };

    let mut found: BTreeMap<(u16, u32), String> = BTreeMap::new();
    let mut cur_pid: u32 = 0;
    let mut cur_cmd = String::new();
    for line in text.lines() {
        let mut chars = line.chars();
        let tag = match chars.next() {
            Some(c) => c,
            None => continue,
        };
        let val = &line[1..];
        match tag {
            'p' => cur_pid = val.parse().unwrap_or(0),
            'c' => cur_cmd = val.to_string(),
            'n' => {
                if let Some(port) = parse_port(val) {
                    found.entry((port, cur_pid)).or_insert_with(|| cur_cmd.clone());
                }
            }
            _ => {}
        }
    }

    let mut cache: BTreeMap<u32, String> = BTreeMap::new();
    let mut entries: Vec<Entry> = Vec::new();
    for ((port, pid), cmd) in found {
        let path = exe_path(pid, &mut cache);
        let name = if path.is_empty() {
            cmd
        } else {
            basename(&path)
        };
        let path = if path.is_empty() {
            "(unknown)".to_string()
        } else {
            path
        };
        entries.push(Entry { port, pid, name, path });
    }
    entries.sort_by(|a, b| a.port.cmp(&b.port).then(a.pid.cmp(&b.pid)));
    entries
}

fn config_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".config/port-doctor/aliases.conf")
}

fn load_aliases() -> BTreeMap<u16, String> {
    let mut map = BTreeMap::new();
    if let Ok(text) = fs::read_to_string(config_path()) {
        for line in text.lines() {
            if let Some((k, v)) = line.split_once('=') {
                if let Ok(port) = k.trim().parse::<u16>() {
                    let alias = v.trim();
                    if !alias.is_empty() {
                        map.insert(port, alias.to_string());
                    }
                }
            }
        }
    }
    map
}

fn save_aliases(map: &BTreeMap<u16, String>) {
    let path = config_path();
    if let Some(dir) = path.parent() {
        let _ = fs::create_dir_all(dir);
    }
    let mut body = String::new();
    for (port, alias) in map {
        body.push_str(&format!("{}={}\n", port, alias));
    }
    let _ = fs::write(path, body);
}

fn cell(out: &mut String, colored: &str, plain_len: usize, width: usize) {
    out.push_str(colored);
    for _ in 0..width.saturating_sub(plain_len) {
        out.push(' ');
    }
    out.push_str("  ");
}

fn print_table(entries: &[Entry], aliases: &BTreeMap<u16, String>) {
    if entries.is_empty() {
        println!("{}No listening TCP ports found.{}", DIM, RESET);
        return;
    }

    let idx_w = entries.len().to_string().len().max(1);
    let port_w = entries
        .iter()
        .map(|e| e.port.to_string().len())
        .max()
        .unwrap_or(4)
        .max("PORT".len());
    let alias_w = entries
        .iter()
        .map(|e| aliases.get(&e.port).map(|a| a.len()).unwrap_or(1))
        .max()
        .unwrap_or(5)
        .max("ALIAS".len());
    let pid_w = entries
        .iter()
        .map(|e| e.pid.to_string().len())
        .max()
        .unwrap_or(3)
        .max("PID".len());
    let name_w = entries
        .iter()
        .map(|e| e.name.len())
        .max()
        .unwrap_or(4)
        .max("NAME".len());

    let mut header = String::new();
    cell(&mut header, &format!("{}{}#{}", BOLD, CYAN, RESET), 1, idx_w);
    cell(&mut header, &format!("{}{}PORT{}", BOLD, CYAN, RESET), 4, port_w);
    cell(&mut header, &format!("{}{}ALIAS{}", BOLD, CYAN, RESET), 5, alias_w);
    cell(&mut header, &format!("{}{}PID{}", BOLD, CYAN, RESET), 3, pid_w);
    cell(&mut header, &format!("{}{}NAME{}", BOLD, CYAN, RESET), 4, name_w);
    header.push_str(&format!("{}{}PATH{}", BOLD, CYAN, RESET));
    println!("{}", header);

    for (i, e) in entries.iter().enumerate() {
        let idx = i + 1;
        let mut row = String::new();
        cell(&mut row, &format!("{}{}{}", DIM, idx, RESET), idx.to_string().len(), idx_w);
        cell(&mut row, &format!("{}{}{}", GREEN, e.port, RESET), e.port.to_string().len(), port_w);
        match aliases.get(&e.port) {
            Some(a) => cell(&mut row, &format!("{}{}{}", MAGENTA, a, RESET), a.len(), alias_w),
            None => cell(&mut row, &format!("{}-{}", DIM, RESET), 1, alias_w),
        }
        cell(&mut row, &format!("{}{}{}", YELLOW, e.pid, RESET), e.pid.to_string().len(), pid_w);
        cell(&mut row, &format!("{}{}{}", WHITE, e.name, RESET), e.name.len(), name_w);
        row.push_str(&format!("{}{}{}", DIM, e.path, RESET));
        println!("{}", row);
    }
}

fn kill_pid(pid: u32) -> Result<(), String> {
    let out = Command::new("/bin/kill")
        .args(["-9", &pid.to_string()])
        .output();
    match out {
        Ok(o) if o.status.success() => Ok(()),
        Ok(o) => Err(String::from_utf8_lossy(&o.stderr).trim().to_string()),
        Err(e) => Err(e.to_string()),
    }
}

fn read_line(prompt: &str) -> String {
    print!("{}", prompt);
    io::stdout().flush().ok();
    let mut s = String::new();
    io::stdin().read_line(&mut s).ok();
    s.trim().to_string()
}

fn banner() {
    println!(
        "{}{} port-doctor {}{}who's on which port — alias it, kill it for good{}",
        BOLD, CYAN, RESET, DIM, RESET
    );
}

fn interactive() {
    loop {
        let entries = scan();
        let aliases = load_aliases();
        println!();
        banner();
        println!();
        print_table(&entries, &aliases);
        println!();
        let choice = read_line(&format!(
            "{}select # (k=kill via menu) · r=refresh · q=quit:{} ",
            BOLD, RESET
        ));
        match choice.as_str() {
            "q" | "quit" | "exit" => break,
            "r" | "" => continue,
            _ => {}
        }
        let idx: usize = match choice.parse() {
            Ok(n) if n >= 1 && n <= entries.len() => n,
            _ => {
                println!("{}not a valid row{}", RED, RESET);
                continue;
            }
        };
        let e = &entries[idx - 1];
        println!(
            "{}port {}{} · pid {}{} · {}{}{} · {}{}",
            DIM, GREEN, e.port, YELLOW, e.pid, WHITE, e.name, DIM, e.path, RESET
        );
        let action = read_line(&format!(
            "{}[k]ill for good · [a]lias/rename · [c]ancel:{} ",
            BOLD, RESET
        ));
        match action.as_str() {
            "k" | "kill" => {
                let ok = read_line(&format!(
                    "{}kill pid {} ({}) on port {} for good? [y/N]:{} ",
                    YELLOW, e.pid, e.name, e.port, RESET
                ));
                if ok == "y" || ok == "Y" {
                    match kill_pid(e.pid) {
                        Ok(()) => println!("{}killed pid {} for good{}", GREEN, e.pid, RESET),
                        Err(msg) => println!("{}could not kill pid {}: {}{}", RED, e.pid, msg, RESET),
                    }
                } else {
                    println!("{}cancelled{}", DIM, RESET);
                }
            }
            "a" | "alias" | "rename" => {
                let mut aliases = aliases;
                let new = read_line(&format!(
                    "{}alias for port {} (blank to clear):{} ",
                    MAGENTA, e.port, RESET
                ));
                if new.is_empty() {
                    aliases.remove(&e.port);
                    println!("{}alias cleared for port {}{}", DIM, e.port, RESET);
                } else {
                    aliases.insert(e.port, new.clone());
                    println!("{}port {} is now \"{}\"{}", GREEN, e.port, new, RESET);
                }
                save_aliases(&aliases);
            }
            _ => println!("{}cancelled{}", DIM, RESET),
        }
    }
}

fn kill_on_port(arg: Option<&String>) {
    let port: u16 = match arg.and_then(|s| s.parse().ok()) {
        Some(p) => p,
        None => {
            println!("{}usage: port-doctor kill <port>{}", RED, RESET);
            return;
        }
    };
    let entries = scan();
    let hits: Vec<&Entry> = entries.iter().filter(|e| e.port == port).collect();
    if hits.is_empty() {
        println!("{}nothing is listening on port {}{}", DIM, port, RESET);
        return;
    }
    for e in hits {
        match kill_pid(e.pid) {
            Ok(()) => println!(
                "{}killed pid {} ({}) on port {} for good{}",
                GREEN, e.pid, e.name, port, RESET
            ),
            Err(msg) => println!(
                "{}could not kill pid {} on port {}: {}{}",
                RED, e.pid, port, msg, RESET
            ),
        }
    }
}

fn set_alias(args: &[String]) {
    if args.len() < 2 {
        println!("{}usage: port-doctor alias <port> <name>   (omit name to clear){}", RED, RESET);
        return;
    }
    let port: u16 = match args[0].parse() {
        Ok(p) => p,
        Err(_) => {
            println!("{}invalid port{}", RED, RESET);
            return;
        }
    };
    let mut aliases = load_aliases();
    let name = args[1..].join(" ");
    if name.is_empty() {
        aliases.remove(&port);
        println!("{}alias cleared for port {}{}", DIM, port, RESET);
    } else {
        aliases.insert(port, name.clone());
        println!("{}port {} is now \"{}\"{}", GREEN, port, name, RESET);
    }
    save_aliases(&aliases);
}

fn help() {
    banner();
    println!();
    println!("{}usage:{}", BOLD, RESET);
    println!("  port-doctor                  interactive: list, alias/rename, kill");
    println!("  port-doctor list             print the port table and exit");
    println!("  port-doctor kill <port>      kill whatever holds <port> for good");
    println!("  port-doctor alias <port> <name>   tag a port with a friendly name");
    println!("  port-doctor help             show this");
    println!();
    println!("{}each row shows: PORT · ALIAS · PID · NAME · full PATH{}", DIM, RESET);
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(|s| s.as_str()) {
        None => interactive(),
        Some("list") | Some("ls") | Some("-l") | Some("--list") => {
            print_table(&scan(), &load_aliases())
        }
        Some("kill") => kill_on_port(args.get(1)),
        Some("alias") | Some("rename") => set_alias(&args[1..]),
        Some("help") | Some("-h") | Some("--help") => help(),
        Some(other) => {
            println!("{}unknown command: {}{}", RED, other, RESET);
            help();
        }
    }
}
