use crate::quartz::{self, Display};
use crate::registry;
use crate::vendor;

const MM_PER_INCH: f64 = 25.4;

pub struct Resolution {
    pub width: usize,
    pub height: usize,
    pub refresh_hz: f64,
}

pub struct Monitor {
    pub name: String,
    pub brand: String,
    pub builtin: bool,
    pub main: bool,
    pub diagonal_inches: f64,
    pub max_resolution: Option<Resolution>,
    pub max_refresh_hz: Option<f64>,
}

pub fn detect() -> Vec<Monitor> {
    let names = registry::product_names();
    quartz::active_displays()
        .iter()
        .map(|display| build(display, &names))
        .collect()
}

fn build(display: &Display, names: &[registry::ProductName]) -> Monitor {
    let brand = vendor::brand(display.vendor);
    Monitor {
        name: name_of(display, names, &brand),
        brand,
        builtin: display.builtin,
        main: display.main,
        diagonal_inches: diagonal_inches(display.width_mm, display.height_mm),
        max_resolution: max_resolution(&display.modes),
        max_refresh_hz: max_refresh_hz(&display.modes),
    }
}

fn name_of(display: &Display, names: &[registry::ProductName], brand: &str) -> String {
    if let Some(name) = registry::lookup(names, display.vendor, display.model) {
        return name.to_string();
    }
    if display.builtin {
        return "Built-in Display".to_string();
    }
    format!("{brand} Display")
}

pub fn diagonal_inches(width_mm: f64, height_mm: f64) -> f64 {
    if width_mm <= 0.0 || height_mm <= 0.0 {
        return 0.0;
    }
    let diagonal = width_mm.hypot(height_mm) / MM_PER_INCH;
    (diagonal * 10.0).round() / 10.0
}

pub fn max_resolution(modes: &[quartz::DisplayMode]) -> Option<Resolution> {
    modes
        .iter()
        .max_by(|a, b| {
            (a.width * a.height)
                .cmp(&(b.width * b.height))
                .then(a.refresh_hz.total_cmp(&b.refresh_hz))
        })
        .map(|mode| Resolution {
            width: mode.width,
            height: mode.height,
            refresh_hz: mode.refresh_hz,
        })
}

pub fn max_refresh_hz(modes: &[quartz::DisplayMode]) -> Option<f64> {
    modes
        .iter()
        .map(|mode| mode.refresh_hz)
        .filter(|hz| *hz > 0.0)
        .max_by(f64::total_cmp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quartz::DisplayMode;

    fn mode(width: usize, height: usize, refresh_hz: f64) -> DisplayMode {
        DisplayMode {
            width,
            height,
            refresh_hz,
        }
    }

    #[test]
    fn converts_the_panel_millimetres_into_the_diagonal_people_shop_for() {
        assert_eq!(diagonal_inches(701.7, 400.5), 31.8);
        assert_eq!(diagonal_inches(301.2, 195.6), 14.1);
    }

    #[test]
    fn reports_zero_inches_when_the_panel_publishes_no_physical_size() {
        assert_eq!(diagonal_inches(0.0, 0.0), 0.0);
        assert_eq!(diagonal_inches(-1.0, 400.0), 0.0);
    }

    #[test]
    fn picks_the_mode_with_the_most_pixels_not_the_fastest_one() {
        let modes = [mode(800, 600, 75.0), mode(3840, 2160, 30.0)];
        let best = max_resolution(&modes).unwrap();
        assert_eq!((best.width, best.height), (3840, 2160));
        assert_eq!(best.refresh_hz, 30.0);
    }

    #[test]
    fn breaks_a_pixel_count_tie_on_the_faster_refresh_rate() {
        let modes = [mode(1920, 1200, 60.0), mode(1920, 1200, 120.0)];
        assert_eq!(max_resolution(&modes).unwrap().refresh_hz, 120.0);
    }

    #[test]
    fn reports_the_fastest_refresh_any_mode_reaches_even_at_a_lower_resolution() {
        let modes = [mode(3840, 2160, 30.0), mode(800, 600, 75.0)];
        assert_eq!(max_refresh_hz(&modes), Some(75.0));
    }

    #[test]
    fn ignores_modes_whose_refresh_rate_the_driver_leaves_at_zero() {
        let modes = [mode(3024, 1964, 0.0)];
        assert_eq!(max_refresh_hz(&modes), None);
        assert!(max_resolution(&modes).is_some());
    }

    #[test]
    fn has_nothing_to_report_when_a_display_exposes_no_modes() {
        assert!(max_resolution(&[]).is_none());
        assert!(max_refresh_hz(&[]).is_none());
    }
}
