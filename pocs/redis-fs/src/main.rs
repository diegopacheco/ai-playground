mod path;
mod store;
mod commands;
mod shell;

fn main() {
    let mut conn = store::connect();
    store::ensure_root(&mut conn);
    shell::run(&mut conn);
}
