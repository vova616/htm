#![feature(conservative_impl_trait)]
#![allow(dead_code)]

pub use self::spatial_pooler::SpatialPooler;
pub use self::sdr_classifier::SDRClassifier;
pub use self::universal_rand::{UniversalRng,UniversalNext};
pub use self::temporal_memory::TemporalMemory;


mod spatial_pooler;
mod sdr_classifier;
mod universal_rand;
mod column;
mod topology;
mod numext;
mod potential_pool;
mod potential_group;
mod dynamic_container;
mod temporal_memory;
mod peekable_while;
mod bucket_container;

extern crate bit_vec;
extern crate rand;
extern crate collect_slice;
extern crate quickersort;
extern crate rayon;
extern crate fnv;

