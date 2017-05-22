extern crate htm;

use htm::{SpatialPooler,SDRClassifier};

fn main() {
    let mut sp = SpatialPooler::new(vec![10], vec![100]);
    sp.potential_radius = 3;
    sp.global_inhibition = true;
    sp.num_active_columns_per_inh_area = 0.02 * sp.num_columns as f64;
    sp.syn_perm_options.active_inc = 0.01;
    sp.syn_perm_options.trim_threshold = 0.005;
    sp.compability_mode = true;

    let mut classifier: SDRClassifier<u8> =
        SDRClassifier::new(vec![0, 1], 0.1, 0.3, sp.num_columns);

  
    println!("Initializing");
    sp.init();


    let mut input = vec![false; sp.num_inputs];

    println!("Training");

    let mut record = 0;
    for i in 0..100 {
        for val in 0..10 {
            for val in &mut input {
                *val = false;
            }
            input[val] = true;

            sp.compute(&input, true);
            sp.winner_columns.sort();

            let r = classifier.compute(record, val, val as u8, &sp.winner_columns[..], true, true);
            if i == 99 {
                println!("value: {}", val);
                for &(ref step, ref probabilities) in &r {
                    println!("{} {:?}",
                             step,
                             probabilities
                                 .iter()
                                 .enumerate()
                                 .max_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap())
                                 .unwrap());
                }
            }


            record += 1;
        }

    }

}