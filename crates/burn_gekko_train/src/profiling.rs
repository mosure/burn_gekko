//! Optional host ranges. No tensor operations, synchronization or scientific settings.
use std::ffi::CStr;

pub(crate) struct Range {
    #[cfg(feature = "profiling")]
    _guard: nvtx::domain::LocalRange<'static>,
}

// The same lexical scope and explicit end calls compile when annotations are disabled.
impl Drop for Range {
    fn drop(&mut self) {}
}

#[inline]
pub(crate) fn range(name: &'static CStr) -> Range {
    #[cfg(feature = "profiling")]
    {
        static DOMAIN: std::sync::OnceLock<nvtx::Domain> = std::sync::OnceLock::new();
        let domain = DOMAIN.get_or_init(|| nvtx::Domain::new(c"burn_gekko"));
        Range {
            _guard: domain.local_range(domain.register_string(name)),
        }
    }
    #[cfg(not(feature = "profiling"))]
    {
        let _ = name;
        Range {}
    }
}
