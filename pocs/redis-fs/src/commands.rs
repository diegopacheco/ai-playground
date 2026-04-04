use redis::Connection;
use std::io::Write;
use std::process::Command;
use crate::path;
use crate::store;

pub fn create(conn: &mut Connection, cwd: &str, path_arg: &str, content: &str) {
    let fpath = path::resolve(cwd, path_arg);
    let parent = path::parent(&fpath);
    let name = path::basename(&fpath);
    store::set_meta(conn, &fpath, "file", content.len());
    store::set_data(conn, &fpath, content);
    store::add_to_dir(conn, &parent, &name);
    println!("created {}", fpath);
}

pub fn cat(conn: &mut Connection, cwd: &str, args: &[&str]) {
    if args.is_empty() {
        println!("usage: cat <path>");
        return;
    }
    let fpath = path::resolve(cwd, args[0]);
    match store::get_data(conn, &fpath) {
        Some(content) => println!("{}", content),
        None => println!("error: {} not found", fpath),
    }
}

pub fn cp(conn: &mut Connection, cwd: &str, args: &[&str]) {
    if args.len() < 2 {
        println!("usage: cp <src> <dst>");
        return;
    }
    let src = path::resolve(cwd, args[0]);
    let dst = path::resolve(cwd, args[1]);
    match store::get_data(conn, &src) {
        Some(data) => {
            let parent = path::parent(&dst);
            let name = path::basename(&dst);
            store::set_meta(conn, &dst, "file", data.len());
            store::set_data(conn, &dst, &data);
            store::add_to_dir(conn, &parent, &name);
            println!("copied {} -> {}", src, dst);
        }
        None => println!("error: {} not found", src),
    }
}

pub fn rm(conn: &mut Connection, cwd: &str, args: &[&str]) {
    if args.is_empty() {
        println!("usage: rm <path>");
        return;
    }
    let fpath = path::resolve(cwd, args[0]);
    let parent = path::parent(&fpath);
    let name = path::basename(&fpath);
    store::del_entry(conn, &fpath);
    store::remove_from_dir(conn, &parent, &name);
    println!("removed {}", fpath);
}

pub fn mkdir(conn: &mut Connection, cwd: &str, args: &[&str]) {
    if args.is_empty() {
        println!("usage: mkdir <path>");
        return;
    }
    let fpath = path::resolve(cwd, args[0]);
    let parent = path::parent(&fpath);
    let name = path::basename(&fpath);
    store::set_meta(conn, &fpath, "dir", 0);
    store::add_to_dir(conn, &parent, &name);
    println!("created directory {}", fpath);
}

pub fn ls(conn: &mut Connection, cwd: &str, args: &[&str]) {
    let fpath = if args.is_empty() {
        cwd.to_string()
    } else {
        path::resolve(cwd, args[0])
    };
    let members = store::list_dir(conn, &fpath);
    if members.is_empty() {
        println!("(empty)");
    } else {
        let mut sorted: Vec<&String> = members.iter().collect();
        sorted.sort();
        for entry in sorted {
            println!("{}", entry);
        }
    }
}

pub fn import(conn: &mut Connection, cwd: &str, args: &[&str]) {
    if args.len() < 2 {
        println!("usage: import <local-path> <virtual-path>");
        return;
    }
    let local_path = args[0];
    let vpath = path::resolve(cwd, args[1]);
    match std::fs::read_to_string(local_path) {
        Ok(content) => {
            let parent = path::parent(&vpath);
            let name = path::basename(&vpath);
            store::set_meta(conn, &vpath, "file", content.len());
            store::set_data(conn, &vpath, &content);
            store::add_to_dir(conn, &parent, &name);
            println!("imported {} -> {}", local_path, vpath);
        }
        Err(e) => println!("error: cannot read {}: {}", local_path, e),
    }
}

pub fn exec(conn: &mut Connection, cwd: &str, args: &[&str]) {
    if args.is_empty() {
        println!("usage: exec <path>");
        return;
    }
    let fpath = path::resolve(cwd, args[0]);
    match store::get_data(conn, &fpath) {
        Some(script) => {
            let tmp = "/tmp/redis-fs-exec.sh";
            let mut file = std::fs::File::create(tmp).unwrap();
            file.write_all(script.as_bytes()).unwrap();
            let output = Command::new("bash").arg(tmp).output().unwrap();
            print!("{}", String::from_utf8_lossy(&output.stdout));
            if !output.stderr.is_empty() {
                eprint!("{}", String::from_utf8_lossy(&output.stderr));
            }
            std::fs::remove_file(tmp).unwrap();
        }
        None => println!("error: {} not found", fpath),
    }
}
