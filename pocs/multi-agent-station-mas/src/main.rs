mod agent;
mod session;
mod ui;
mod input;
mod app;

use anyhow::Result;
use app::App;

fn main() -> Result<()> {
    let mut app = App::new(24, 80);
    app.run()
}
