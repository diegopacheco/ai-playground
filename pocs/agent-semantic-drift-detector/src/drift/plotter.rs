use crate::persistence::models::DriftReport;

const PLOT_WIDTH: usize = 60;
const PLOT_HEIGHT: usize = 15;

pub fn plot_drift(reports: &[DriftReport]) {
    if reports.is_empty() {
        println!("No drift data to plot yet. Run the probe at least twice.");
        return;
    }

    println!("\n=== Semantic Drift Plot (Cosine Similarity vs Baseline) ===\n");

    let min_sim = reports.iter()
        .map(|r| r.cosine_similarity)
        .fold(f64::INFINITY, f64::min)
        .min(0.5);
    let max_sim = 1.0;

    let mut grid = vec![vec![' '; PLOT_WIDTH]; PLOT_HEIGHT];

    for (i, report) in reports.iter().enumerate() {
        let x = if reports.len() == 1 {
            PLOT_WIDTH / 2
        } else {
            i * (PLOT_WIDTH - 1) / (reports.len() - 1)
        };

        let normalized = (report.cosine_similarity - min_sim) / (max_sim - min_sim);
        let y = ((1.0 - normalized) * (PLOT_HEIGHT - 1) as f64) as usize;
        let y = y.min(PLOT_HEIGHT - 1);

        if report.drift_detected {
            grid[y][x] = 'X';
        } else {
            grid[y][x] = '*';
        }
    }

    for (row_idx, row) in grid.iter().enumerate() {
        let sim_val = max_sim - (row_idx as f64 / (PLOT_HEIGHT - 1) as f64) * (max_sim - min_sim);
        print!("{:5.2} | ", sim_val);
        for c in row {
            print!("{}", c);
        }
        println!();
    }

    print!("      +");
    for _ in 0..PLOT_WIDTH {
        print!("-");
    }
    println!();

    println!("       Oldest{:>width$}Latest", "", width = PLOT_WIDTH - 10);
    println!();
    println!("  Legend: * = stable, X = drift detected (similarity < 0.75)");
    println!();

    println!("=== Drift Report ===\n");
    println!("{:<22} {:>12} {:>10}", "Date", "Similarity", "Drifted?");
    println!("{}", "-".repeat(46));
    for report in reports {
        println!(
            "{:<22} {:>12.4} {:>10}",
            report.date,
            report.cosine_similarity,
            if report.drift_detected { "YES" } else { "no" }
        );
    }
    println!();
}
