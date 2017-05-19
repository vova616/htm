#![feature(conservative_impl_trait)]
#![allow(dead_code)]

mod spatial_pooler;
mod column;
mod topology;
mod numext;
mod potential_pool;
mod universal_rand;

extern crate bit_vec;
extern crate rand;
extern crate collect_slice;
extern crate quickersort;
extern crate rayon;
extern crate time;

use rand::*;
use spatial_pooler::SpatialPooler;
use universal_rand::*;
use time::PreciseTime;


fn main() { 
    let mut sp = SpatialPooler::new(vec![32,32], vec![64,64]);
    sp.potential_radius = sp.num_inputs as i32;
    sp.global_inhibition = false;
    sp.num_active_columns_per_inh_area = 0.02 * sp.num_columns as f64;
    sp.syn_perm_options.active_inc = 0.01;
    sp.syn_perm_options.trim_threshold = 0.005;
    sp.compability_mode = true;

    {
        print!("Initializing");
        let start = PreciseTime::now();
        sp.init();
        println!(": {:?}", start.to(PreciseTime::now()));
    }

    let mut rnd = UniversalRng::from_seed([42,0,0,0]);
    let mut input = vec![false; sp.num_inputs];
    let mut active_array = vec![false; sp.num_inputs];

    for _ in 0..10
    {
        for val in &mut input {
            *val = rnd.next_uv_int(2) == 1;
        }

        print!("Computing");
        let start = PreciseTime::now();
        sp.compute(&input, &mut active_array, true);
        println!(": {:?}", start.to(PreciseTime::now()).num_microseconds().unwrap() as f64 / 1000.0);
        //println!("{:?}", sp.overlaps);
        //println!("{:?}", sp.winner_columns);
    }

}
