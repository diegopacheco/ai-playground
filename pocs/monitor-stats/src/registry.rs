use std::process::Command;

pub struct ProductName {
    pub vendor: u32,
    pub model: u32,
    pub name: String,
}

pub fn product_names() -> Vec<ProductName> {
    let output = Command::new("ioreg")
        .args(["-lw0", "-r", "-c", "IOMobileFramebuffer"])
        .output();
    match output {
        Ok(out) if out.status.success() => parse(&String::from_utf8_lossy(&out.stdout)),
        _ => Vec::new(),
    }
}

pub fn parse(ioreg_dump: &str) -> Vec<ProductName> {
    let mut out = Vec::new();
    for block in ioreg_dump.split("\"ProductAttributes\"=").skip(1) {
        let block = block.split('}').next().unwrap_or_default();
        let name = string_field(block, "ProductName");
        let vendor = number_field(block, "LegacyManufacturerID");
        let model = number_field(block, "ProductID");
        if let (Some(name), Some(vendor), Some(model)) = (name, vendor, model) {
            out.push(ProductName {
                vendor,
                model,
                name,
            });
        }
    }
    out
}

fn field<'a>(block: &'a str, key: &str) -> Option<&'a str> {
    let marker = format!("\"{key}\"=");
    let start = block.find(&marker)? + marker.len();
    Some(&block[start..])
}

fn string_field(block: &str, key: &str) -> Option<String> {
    let rest = field(block, key)?.strip_prefix('"')?;
    let value = rest.split('"').next()?.trim();
    (!value.is_empty()).then(|| value.to_string())
}

fn number_field(block: &str, key: &str) -> Option<u32> {
    let rest = field(block, key)?;
    let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

pub fn lookup(names: &[ProductName], vendor: u32, model: u32) -> Option<&str> {
    names
        .iter()
        .find(|entry| entry.vendor == vendor && entry.model == model)
        .map(|entry| entry.name.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    const DUMP: &str = r#"
    "DisplayAttributes" = {"TiledDisplayInfo"={},"ProductAttributes"={"ManufacturerID"="00-10-fa","ProductID"=55204023186497,"LegacyManufacturerID"=1552},"DisplayAllocation"={"ExtraPipes"=0}}
    "DisplayAttributes" = {"MaximumRefreshRate"=75,"ProductAttributes"={"YearOfManufacture"=2021,"ManufacturerID"="SAM","SerialNumber"=809646411,"ProductName"="LS32A70","LegacyManufacturerID"=19501,"ProductID"=29029,"WeekOfManufacture"=50},"MaxVerticalImageSize"=40}
    "#;

    #[test]
    fn reads_the_panel_model_name_out_of_an_ioreg_dump() {
        let names = parse(DUMP);
        assert_eq!(names.len(), 1);
        assert_eq!(names[0].name, "LS32A70");
        assert_eq!(names[0].vendor, 19501);
        assert_eq!(names[0].model, 29029);
    }

    #[test]
    fn skips_panels_that_publish_no_product_name() {
        let names = parse(DUMP);
        assert!(names.iter().all(|entry| entry.vendor != 1552));
    }

    #[test]
    fn matches_a_panel_only_when_both_vendor_and_model_agree() {
        let names = parse(DUMP);
        assert_eq!(lookup(&names, 19501, 29029), Some("LS32A70"));
        assert_eq!(lookup(&names, 19501, 1), None);
        assert_eq!(lookup(&names, 1, 29029), None);
    }

    #[test]
    fn returns_nothing_for_an_empty_or_unparsable_dump() {
        assert!(parse("").is_empty());
        assert!(parse("garbage without any attributes").is_empty());
    }
}
