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
modulo_signed_ext_impl! { i8 i16 i32 i64 usize isize}


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
