const BRANDS: &[(&str, &str)] = &[
    ("ACI", "ASUS"),
    ("ACR", "Acer"),
    ("AGO", "AOC"),
    ("AOC", "AOC"),
    ("API", "Acer"),
    ("APP", "Apple"),
    ("AUO", "AU Optronics"),
    ("AUS", "ASUS"),
    ("BNQ", "BenQ"),
    ("BOE", "BOE"),
    ("CMN", "Chi Mei"),
    ("CMO", "Chi Mei"),
    ("DEL", "Dell"),
    ("ENC", "EIZO"),
    ("EIZ", "EIZO"),
    ("GBT", "Gigabyte"),
    ("GSM", "LG"),
    ("HPN", "HP"),
    ("HWP", "HP"),
    ("HSD", "Hannspree"),
    ("IVM", "Iiyama"),
    ("LEN", "Lenovo"),
    ("LGD", "LG Display"),
    ("MSI", "MSI"),
    ("NEC", "NEC"),
    ("PHL", "Philips"),
    ("PLN", "Planar"),
    ("SAM", "Samsung"),
    ("SDC", "Samsung Display"),
    ("SEC", "Samsung"),
    ("SHP", "Sharp"),
    ("SNY", "Sony"),
    ("VSC", "ViewSonic"),
];

pub fn pnp_code(vendor: u32) -> Option<String> {
    if vendor == 0 || vendor > 0x7FFF {
        return None;
    }
    let code = vendor as u16;
    let letters = [(code >> 10) & 0x1F, (code >> 5) & 0x1F, code & 0x1F];
    if letters.iter().any(|&l| !(1..=26).contains(&l)) {
        return None;
    }
    Some(
        letters
            .iter()
            .map(|&l| (b'A' + l as u8 - 1) as char)
            .collect(),
    )
}

pub fn brand(vendor: u32) -> String {
    let Some(code) = pnp_code(vendor) else {
        return "Unknown".to_string();
    };
    BRANDS
        .iter()
        .find(|(pnp, _)| *pnp == code)
        .map(|(_, name)| (*name).to_string())
        .unwrap_or(code)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_the_five_bit_packed_edid_letters() {
        assert_eq!(pnp_code(19501).as_deref(), Some("SAM"));
        assert_eq!(pnp_code(0x0610).as_deref(), Some("APP"));
        assert_eq!(pnp_code(0x1E6D).as_deref(), Some("GSM"));
    }

    #[test]
    fn rejects_vendor_ids_that_are_not_valid_edid_letters() {
        assert_eq!(pnp_code(0), None);
        assert_eq!(pnp_code(0xFFFF), None);
        assert_eq!(pnp_code(0x0400), None);
    }

    #[test]
    fn maps_a_known_pnp_code_to_the_brand_people_recognise() {
        assert_eq!(brand(19501), "Samsung");
        assert_eq!(brand(0x0610), "Apple");
    }

    #[test]
    fn falls_back_to_the_raw_code_when_the_brand_is_not_in_the_table() {
        assert_eq!(brand(0x5A5A), "VRZ");
    }

    #[test]
    fn reports_unknown_when_the_vendor_id_cannot_be_decoded() {
        assert_eq!(brand(0), "Unknown");
    }
}
