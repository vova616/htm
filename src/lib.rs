#![feature(conservative_impl_trait)]
#![allow(dead_code)]



pub use self::util::{UniversalRng,UniversalNext};
pub use self::algo::{TemporalMemory,SDRClassifier,Cell,Synapse,Segment, SpatialPooler};
pub use self::encoder::{ScalarEncoder,DeltaEncoder,AdaptiveScalarEncoder};

mod encoder;
mod util;
mod algo;

extern crate bit_vec;
extern crate rand;
extern crate collect_slice;
extern crate quickersort;
extern crate rayon;
extern crate fnv;

#[macro_use]
extern crate log;
