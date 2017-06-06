pub trait ModuloSignedExt {
    fn modulo(self, n: Self) -> Self;
}
macro_rules! modulo_signed_ext_impl {
    ($($t:ty)*) => ($(
        impl ModuloSignedExt for $t {
            #[inline]
            fn modulo(self, n: Self) -> Self {
                (self % n + n) % n
            }
        }
    )*)
}
modulo_signed_ext_impl! { i8 i16 i32 i64 usize isize f32 f64}


pub trait ClipExt {
    fn clip(self, min: Self, max: Self) -> Self;
}
macro_rules! clip_ext_impl {
    ($($t:ty)*) => ($(
        impl ClipExt for $t {
            #[inline]
            fn clip(self, min: Self, max: Self) -> Self {
                if self < min {
                    min
                } else if self > max {
                    max
                } else {
                    self
                }
            }
        }
    )*)
}
clip_ext_impl! { i8 i16 i32 i64 usize isize f32 f64}

pub trait NaNExt {
    fn is_nan_generic(self) -> bool;
}
macro_rules! nan_ext_impl {
    ($($t:ty)*) => ($(
        impl NaNExt for $t {
            #[inline]
            fn is_nan_generic(self) -> bool {
                self.is_nan()
            }
        }
    )*)
}
nan_ext_impl! {f32 f64}

macro_rules! nan_empty_ext_impl {
    ($($t:ty)*) => ($(
        impl NaNExt for $t {
            #[inline]
             fn is_nan_generic(self) -> bool {
                return false
            }
        }
    )*)
}
nan_empty_ext_impl! { i8 u8 i16 u16 i32 u32 i64 u64 usize isize }