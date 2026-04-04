use redis::Connection;
use std::io::{self, BufRead, Write};
use crate::commands;
use crate::path;

fn is_tty() -> bool {
    unsafe { libc_isatty(0) != 0 }
}

unsafe extern "C" {
    #[link_name = "isatty"]
    fn libc_isatty(fd: i32) -> i32;
}

fn prompt(cwd: &str) {
    print!("redis-fs:{} > ", cwd);
    io::stdout().flush().unwrap();
}

pub fn run(conn: &mut Connection) {
    let mut cwd = "/".to_string();
    let stdin = io::stdin();
    let tty = is_tty();

    if tty {
        prompt(&cwd);
    }

    for line in stdin.lock().lines() {
        let line = line.unwrap();
        let line = line.trim().to_string();
        if line.is_empty() {
            if tty {
                prompt(&cwd);
            }
            continue;
        }

        let parts: Vec<&str> = line.splitn(2, ' ').collect();
        let cmd = parts[0];
        let rest = if parts.len() > 1 { parts[1] } else { "" };
        let args: Vec<&str> = rest.split_whitespace().collect();

        match cmd {
            "create" => {
                let create_parts: Vec<&str> = rest.splitn(2, ' ').collect();
                if create_parts.len() == 2 {
                    commands::create(conn, &cwd, create_parts[0], create_parts[1]);
                } else if create_parts.len() == 1 && !create_parts[0].is_empty() {
                    commands::create(conn, &cwd, create_parts[0], "");
                } else {
                    println!("usage: create <path> <content>");
                }
            }
            "cat" => commands::cat(conn, &cwd, &args),
            "cp" => commands::cp(conn, &cwd, &args),
            "rm" => commands::rm(conn, &cwd, &args),
            "mkdir" => commands::mkdir(conn, &cwd, &args),
            "ls" => commands::ls(conn, &cwd, &args),
            "cd" => {
                if args.is_empty() {
                    cwd = "/".to_string();
                } else {
                    cwd = path::resolve(&cwd, args[0]);
                }
            }
            "pwd" => println!("{}", cwd),
            "import" => commands::import(conn, &cwd, &args),
            "exec" => commands::exec(conn, &cwd, &args),
            "exit" => break,
            _ => println!("unknown command: {}", cmd),
        }

        if tty {
            prompt(&cwd);
        }
    }

    if tty {
        println!();
    }
}
