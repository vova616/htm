extern crate htm;
extern crate time;
extern crate rand;

use htm::{SpatialPooler,UniversalRng,UniversalNext};
use time::PreciseTime;

fn main() {
    let mut sp = SpatialPooler::new(vec![32, 32], vec![64, 64]);
    sp.potential_radius = sp.num_inputs as i32;
    sp.global_inhibition = true;
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

    let mut rnd = UniversalRng::from_seed([42, 0, 0, 0]);
    let mut input = vec![false; sp.num_inputs];

    let mut record = 0;
    for i in 0..10 {
        for val in &mut input {
            *val = rnd.next_uv_int(2) == 1;
        }

        sp.compute(&input, true);
        //println!(": {:?}", start.to(PreciseTime::now()).num_microseconds().unwrap() as f64 / 1000.0);
        //println!("{:?}", sp.overlaps);
        sp.winner_columns.sort();
        println!("{:?}", sp.winner_columns);
    }   
}