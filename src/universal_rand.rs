/// An Xorshift[1] random number
/// generator.
///
/// The Xorshift algorithm is not suitable for cryptographic purposes
/// but is very fast. If you do not know for sure that it fits your
/// requirements, use a more secure one such as `IsaacRng` or `OsRng`.
///
/// [1]: Marsaglia, George (July 2003). ["Xorshift
/// RNGs"](http://www.jstatsoft.org/v08/i14/paper). *Journal of
/// Statistical Software*. Vol. 8 (Issue 14).
///use std::num::Wrapping as w;

use rand::{Rand,SeedableRng,Rng};

#[allow(missing_copy_implementations)]
#[derive(Clone, Debug)]
pub struct UniversalRng {
    seed: u64,
}

impl UniversalRng {
    /// Creates a new XorShiftRng instance which is not seeded.
    ///
    /// The initial values of this RNG are constants, so all generators created
    /// by this function will yield the same stream of random numbers. It is
    /// highly recommended that this is created through `SeedableRng` instead of
    /// this function
    pub fn new_unseeded() -> UniversalRng {
        UniversalRng { seed: 0x193a6754 }
    }

    pub fn from_seed(seed: [u32; 4]) -> UniversalRng {
        assert!(!seed.iter().all(|&x| x == 0),
                "UniversalRng::from_seed called with an all zero seed.");

        UniversalRng { seed: seed[0] as u64 + seed[1] as u64 + seed[2] as u64 + seed[3] as u64 }
    }

    pub fn reseed(&mut self, seed: [u32; 4]) {
        assert!(!seed.iter().all(|&x| x == 0),
                "UniversalRng.reseed called with an all zero seed.");

        self.seed = seed[0] as u64 + seed[1] as u64 + seed[2] as u64 + seed[3] as u64;
    }
}


pub trait UniversalNext {
    fn next_uv_int(&mut self, bound: i32) -> i32;
}

impl<R: Rng> UniversalNext for R {
    #[inline]
    fn next_uv_int(&mut self, bound: i32) -> i32 {
        if bound <= 0 {
            panic!("Bad bound");
        }
        let r = self.next_u32() as i32;
        let m = bound - 1;
        if bound & m == 0 {
            // i.e., bound is a power of 2
            ((bound as i64 * r as i64) >> 31) as i32
        } else {
            r % bound
        }
    }
}

impl Rng for UniversalRng {
    #[inline]
    fn next_u32(&mut self) -> u32 {
        let mut x = self.seed;
        x ^= (x << 21) & 0xffffffffffffffff;
        x ^= (x >> 35) & 0xffffffffffffffff;
        x ^= (x << 4) & 0xffffffffffffffff;
        self.seed = x;
        x &= (1 << 31) - 1;

        x as u32
    }

    #[inline]
    fn next_f32(&mut self) -> f32 {
        self.next_uv_int(10000) as f32 * 0.0001
    }

    #[inline]
    fn next_f64(&mut self) -> f64 {
        self.next_uv_int(10000) as f64 * 0.0001
    }
}

impl SeedableRng<[u32; 4]> for UniversalRng {
    /// Reseed an UniversalRng. This will panic if `seed` is entirely 0.
    fn reseed(&mut self, seed: [u32; 4]) {
        self.reseed(seed);
    }

    /// Create a new UniversalRng. This will panic if `seed` is entirely 0.
    fn from_seed(seed: [u32; 4]) -> UniversalRng {
        UniversalRng::from_seed(seed)
    }
}

impl Rand for UniversalRng {
    fn rand<R: Rng>(rng: &mut R) -> UniversalRng {
        let mut tuple: (u32, u32, u32, u32) = rng.gen();
        while tuple == (0, 0, 0, 0) {
            tuple = rng.gen();
        }
        let (x, y, z, w) = tuple;
        UniversalRng { seed: x as u64 + y as u64 + z as u64 + w as u64 }
    }
}
