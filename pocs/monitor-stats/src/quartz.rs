use std::ffi::c_void;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct CGSize {
    pub width: f64,
    pub height: f64,
}

pub type CGDirectDisplayID = u32;
pub type CGDisplayModeRef = *const c_void;
type CFArrayRef = *const c_void;

#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {
    fn CGGetActiveDisplayList(
        max_displays: u32,
        displays: *mut CGDirectDisplayID,
        count: *mut u32,
    ) -> i32;
    fn CGDisplayScreenSize(display: CGDirectDisplayID) -> CGSize;
    fn CGDisplayIsBuiltin(display: CGDirectDisplayID) -> i32;
    fn CGDisplayIsMain(display: CGDirectDisplayID) -> i32;
    fn CGDisplayVendorNumber(display: CGDirectDisplayID) -> u32;
    fn CGDisplayModelNumber(display: CGDirectDisplayID) -> u32;
    fn CGDisplayCopyAllDisplayModes(
        display: CGDirectDisplayID,
        options: *const c_void,
    ) -> CFArrayRef;
    fn CGDisplayModeGetPixelWidth(mode: CGDisplayModeRef) -> usize;
    fn CGDisplayModeGetPixelHeight(mode: CGDisplayModeRef) -> usize;
    fn CGDisplayModeGetRefreshRate(mode: CGDisplayModeRef) -> f64;
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFArrayGetCount(array: CFArrayRef) -> isize;
    fn CFArrayGetValueAtIndex(array: CFArrayRef, index: isize) -> *const c_void;
    fn CFRelease(cf: *const c_void);
}

const MAX_DISPLAYS: u32 = 32;

pub struct DisplayMode {
    pub width: usize,
    pub height: usize,
    pub refresh_hz: f64,
}

pub struct Display {
    pub builtin: bool,
    pub main: bool,
    pub vendor: u32,
    pub model: u32,
    pub width_mm: f64,
    pub height_mm: f64,
    pub modes: Vec<DisplayMode>,
}

pub fn active_displays() -> Vec<Display> {
    let mut ids = [0 as CGDirectDisplayID; MAX_DISPLAYS as usize];
    let mut count: u32 = 0;
    let status = unsafe { CGGetActiveDisplayList(MAX_DISPLAYS, ids.as_mut_ptr(), &mut count) };
    if status != 0 {
        return Vec::new();
    }
    ids[..count as usize]
        .iter()
        .map(|&id| describe(id))
        .collect()
}

fn describe(id: CGDirectDisplayID) -> Display {
    let size = unsafe { CGDisplayScreenSize(id) };
    Display {
        builtin: unsafe { CGDisplayIsBuiltin(id) } != 0,
        main: unsafe { CGDisplayIsMain(id) } != 0,
        vendor: unsafe { CGDisplayVendorNumber(id) },
        model: unsafe { CGDisplayModelNumber(id) },
        width_mm: size.width,
        height_mm: size.height,
        modes: modes(id),
    }
}

fn modes(id: CGDirectDisplayID) -> Vec<DisplayMode> {
    let array = unsafe { CGDisplayCopyAllDisplayModes(id, std::ptr::null()) };
    if array.is_null() {
        return Vec::new();
    }
    let count = unsafe { CFArrayGetCount(array) };
    let mut out = Vec::with_capacity(count.max(0) as usize);
    for index in 0..count {
        let mode = unsafe { CFArrayGetValueAtIndex(array, index) };
        if mode.is_null() {
            continue;
        }
        out.push(DisplayMode {
            width: unsafe { CGDisplayModeGetPixelWidth(mode) },
            height: unsafe { CGDisplayModeGetPixelHeight(mode) },
            refresh_hz: unsafe { CGDisplayModeGetRefreshRate(mode) },
        });
    }
    unsafe { CFRelease(array) };
    out
}
