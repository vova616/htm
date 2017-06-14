extern crate htm;
extern crate rand;

use htm::*;
use rand::Rng;

pub fn assert_approx_eq<T : std::ops::Sub<Output=T> + std::fmt::Debug + std::cmp::PartialOrd + Copy>(x: &[T], y: &[T], eps: T) {
    for (v1,v2) in x.iter().zip(y.iter()) {
        let abs = if *v1 > *v2 {
            *v1 - *v2
        } else {
            *v2 - *v1
        };
        assert!(abs < eps,
            "assertion failed: `(left !== right)` \
                        (left: `{:?}`, right: `{:?}`, expect diff: `{:?}`, real diff: `{:?}`)",
                *v1, *v2, eps, abs);
    }
}   

#[macro_export]
macro_rules! assert_approx_eq {
    ($a:expr, $b:expr) => ({
        let eps = 1.0e-6;
        let (a, b) = (&$a, &$b);
        assert!((*a - *b).abs() < eps,
                "assertion failed: `(left !== right)` \
                           (left: `{:?}`, right: `{:?}`, expect diff: `{:?}`, real diff: `{:?}`)",
                 *a, *b, eps, (*a - *b).abs());
    });
    ($a:expr, $b:expr, $eps:expr) => ({
        let (a, b) = (&$a, &$b);
        assert!((*a - *b).abs() < $eps,
                "assertion failed: `(left !== right)` \
                           (left: `{:?}`, right: `{:?}`, expect diff: `{:?}`, real diff: `{:?}`)",
                 *a, *b, $eps, (*a - *b).abs());
    })  
}

pub fn create_tm() -> TemporalMemory {
    create_tm_cells(4)
}

pub fn create_tm_cells(cells: u32) -> TemporalMemory {
    create_tm_custom(32, cells)
}

pub fn create_tm_custom(columns: u32, cells: u32) -> TemporalMemory {
    let mut tm = TemporalMemory::new(columns, cells);
    tm.activation_threshold = 3;
    tm.initial_permanence = 0.21;
    tm.connected_permanence = 0.5;
    tm.min_threshold = 2;
    tm.max_new_synapse_count = 3;
    tm.permanence_increment = 0.10;
    tm.permanence_decrement = 0.10;
    tm.predicted_segment_decrement = 0.0;
    tm.rand = UniversalRng::from_seed([42,0,0,0]);

    tm
}

#[test]
pub fn test_activate_correctly_predictive_cells() {
    let mut tm = create_tm();
    let previous_active_columns = [0];
    let active_columns = [1];

    let expected_active_cells = [ tm.get_cell(4) ];
    let mut segment = tm.create_segment(tm.get_cell(4));
    segment.create_synapse(tm.get_cell(0), 0.5);
    segment.create_synapse(tm.get_cell(1), 0.5);
    segment.create_synapse(tm.get_cell(2), 0.5);
    segment.create_synapse(tm.get_cell(3), 0.5);
    tm.add_segment(segment);

    tm.compute(&previous_active_columns, true);

    {
        let predictive_cells = tm.get_predictive_cells();
        assert_eq!(expected_active_cells.len(), predictive_cells.len());
        for cell in &expected_active_cells {
            assert_eq!(true, predictive_cells.contains(&cell));
        }
    }

    tm.compute(&active_columns, true);
    assert_eq!(expected_active_cells.len(), tm.active_cells.len());
    for cell in &expected_active_cells {
        assert_eq!(true, tm.active_cells.contains(&cell));
    }
}


#[test]
pub fn test_burst_unpredicted_columns() {
    let mut tm = create_tm();
    let active_columns = [0];
    let bursting_cells = [ tm.get_cell(0), tm.get_cell(1), tm.get_cell(2), tm.get_cell(3) ];
    
    tm.compute(&active_columns, true);
    assert_eq!(bursting_cells.len(), tm.active_cells.len());
    for cell in &bursting_cells {
        assert_eq!(true, tm.active_cells.contains(&cell));
    }
}


#[test]
pub fn test_zero_active_columns() {
    let mut tm = create_tm();
    
    let previous_active_columns = [0];

    let expected_active_cells = [ tm.get_cell(4) ];
    let mut segment = tm.create_segment(tm.get_cell(4));
    segment.create_synapse(tm.get_cell(0), 0.5);
    segment.create_synapse(tm.get_cell(1), 0.5);
    segment.create_synapse(tm.get_cell(2), 0.5);
    segment.create_synapse(tm.get_cell(3), 0.5);
    tm.add_segment(segment);

    tm.compute(&previous_active_columns, true);

    assert_eq!(false, tm.active_cells.len() == 0);
    assert_eq!(false, tm.winner_cells.len() == 0);
    assert_eq!(false, tm.get_predictive_cells().len() == 0);
    
    tm.compute(&[], true);
    assert_eq!(true, tm.active_cells.len() == 0);
    assert_eq!(true, tm.winner_cells.len() == 0);
    assert_eq!(true, tm.get_predictive_cells().len() == 0);
}


#[test]
pub fn test_predicted_active_cells_are_always_winners() {
    let mut tm = create_tm();
    let previous_active_columns = [0];
    let active_columns = [1];
    let previous_active_cells = [ tm.get_cell(0), tm.get_cell(1), tm.get_cell(2), tm.get_cell(3) ];
    let expected_winner_cells = [ tm.get_cell(4), tm.get_cell(6) ];
    
    {
    let mut seg = tm.create_segment(expected_winner_cells[0]);
    seg.create_synapse(previous_active_cells[0], 0.5);
    seg.create_synapse(previous_active_cells[1], 0.5);
    seg.create_synapse(previous_active_cells[2], 0.5);
    tm.add_segment(seg);
    }
    
    {
    let mut seg = tm.create_segment(expected_winner_cells[1]);
    seg.create_synapse(previous_active_cells[0], 0.5);
    seg.create_synapse(previous_active_cells[1], 0.5);
    seg.create_synapse(previous_active_cells[2], 0.5);
    tm.add_segment(seg);
    }

   tm.compute(&previous_active_columns, false); // learn=false
   tm.compute(&active_columns, false); // learn=false
    
   assert_eq!(expected_winner_cells.len(), tm.winner_cells.len());
   for cell in &expected_winner_cells {
       assert_eq!(true, tm.winner_cells.contains(&cell));
   }
    
}

#[test]
pub fn test_reinforced_correctly_active_segments() {
    let mut tm = create_tm();
    tm.initial_permanence = 0.2;
    tm.max_new_synapse_count = 4;
    tm.permanence_decrement = 0.08;
    tm.predicted_segment_decrement = 0.02;

    let previous_active_columns = [0];
    let active_columns = [1];
    let previous_active_cells = [ tm.get_cell(0), tm.get_cell(1), tm.get_cell(2), tm.get_cell(3) ];
    let active_cell = tm.get_cell(5);
    
    {
    let mut seg = tm.create_segment(active_cell);
    seg.create_synapse(previous_active_cells[0], 0.5);
    seg.create_synapse(previous_active_cells[1], 0.5);
    seg.create_synapse(previous_active_cells[2], 0.5);
    seg.create_synapse(tm.get_cell(81), 0.5);
    tm.add_segment(seg);
    }

    tm.compute(&previous_active_columns, true);
    tm.compute(&active_columns, true);

    let seg = &tm.get_segments(active_cell)[0];
    assert_approx_eq!(0.6, seg.synapses[0].permanence, 0.1);
    assert_approx_eq!(0.6, seg.synapses[1].permanence, 0.1);
    assert_approx_eq!(0.6, seg.synapses[2].permanence, 0.1);
    assert_approx_eq!(0.42, seg.synapses[3].permanence, 0.001);
}

#[test]
pub fn test_reinforced_selected_matching_segment_in_bursting_column() {
    let mut tm = create_tm();
    tm.permanence_decrement = 0.08;

    let previous_active_columns = [0];
    let active_columns = [1];
    let previous_active_cells = [ tm.get_cell(0), tm.get_cell(1), tm.get_cell(2), tm.get_cell(3) ];
    let bursting_cells = [ tm.get_cell(4), tm.get_cell(5)];
    
    {
    let mut seg = tm.create_segment(bursting_cells[0]);
    seg.create_synapse(previous_active_cells[0], 0.3);
    seg.create_synapse(previous_active_cells[1], 0.3);
    seg.create_synapse(previous_active_cells[2], 0.3);
    seg.create_synapse(tm.get_cell(81), 0.3);
    tm.add_segment(seg);
    }

    {
    let mut seg = tm.create_segment(bursting_cells[1]);
    seg.create_synapse(previous_active_cells[0], 0.3);
    seg.create_synapse(previous_active_cells[1], 0.3);
    seg.create_synapse(tm.get_cell(81), 0.3);
    tm.add_segment(seg);
    }

    tm.compute(&previous_active_columns, true);
    tm.compute(&active_columns, true);

    let seg = &tm.get_segments(bursting_cells[0])[0];
    assert_approx_eq!(0.4, seg.synapses[0].permanence, 0.01);
    assert_approx_eq!(0.4, seg.synapses[1].permanence, 0.01);
    assert_approx_eq!(0.4, seg.synapses[2].permanence, 0.01);
    assert_approx_eq!(0.22, seg.synapses[3].permanence, 0.001);
}


#[test]
pub fn test_no_change_to_non_selected_matching_segments_in_bursting_column() {
    let mut tm = create_tm();
    tm.permanence_decrement = 0.08;

    let previous_active_columns = [0];
    let active_columns = [1];
    let previous_active_cells = [ tm.get_cell(0), tm.get_cell(1), tm.get_cell(2), tm.get_cell(3) ];
    let bursting_cells = [ tm.get_cell(4), tm.get_cell(5)];
    
    {
    let mut seg = tm.create_segment(bursting_cells[0]);
    seg.create_synapse(previous_active_cells[0], 0.3);
    seg.create_synapse(previous_active_cells[1], 0.3);
    seg.create_synapse(previous_active_cells[2], 0.3);
    seg.create_synapse(tm.get_cell(81), 0.3);
    tm.add_segment(seg);
    }

    {
    let mut seg = tm.create_segment(bursting_cells[1]);
    seg.create_synapse(previous_active_cells[0], 0.3);
    seg.create_synapse(previous_active_cells[1], 0.3);
    seg.create_synapse(tm.get_cell(81), 0.3);
    tm.add_segment(seg);
    }

    tm.compute(&previous_active_columns, true);
    tm.compute(&active_columns, true);

    let seg = &tm.get_segments(bursting_cells[1])[0];
    assert_approx_eq!(0.3, seg.synapses[0].permanence, 0.01);
    assert_approx_eq!(0.3, seg.synapses[1].permanence, 0.01);
    assert_approx_eq!(0.3, seg.synapses[2].permanence, 0.001);
}   

#[test]
pub fn test_no_change_to_matching_segments_in_predicted_active_column() {
    let mut tm = create_tm();

    let previous_active_columns = [0];
    let active_columns = [1];
    let previous_active_cells = [ tm.get_cell(0), tm.get_cell(1), tm.get_cell(2), tm.get_cell(3) ];
    let expected_active_cells = [ tm.get_cell(4) ];
    let other_bursting_cell = tm.get_cell(5);
    
    {
    let mut seg = tm.create_segment(expected_active_cells[0]);
    seg.create_synapse(previous_active_cells[0], 0.5);
    seg.create_synapse(previous_active_cells[1], 0.5);
    seg.create_synapse(previous_active_cells[2], 0.5);
    seg.create_synapse(previous_active_cells[3], 0.5);
    tm.add_segment(seg);
    }

    {
    let mut seg = tm.create_segment(expected_active_cells[0]);
    seg.create_synapse(previous_active_cells[0], 0.3);
    seg.create_synapse(previous_active_cells[1], 0.3);
    tm.add_segment(seg);
    }

    {
    let mut seg = tm.create_segment(other_bursting_cell);
    seg.create_synapse(previous_active_cells[0], 0.3);
    seg.create_synapse(previous_active_cells[1], 0.3);
    tm.add_segment(seg);
    }

    tm.compute(&previous_active_columns, true);

    {
        let predictive_cells = tm.get_predictive_cells();
        assert_eq!(expected_active_cells.len(), predictive_cells.len());
        for cell in &expected_active_cells {
            assert_eq!(true, predictive_cells.contains(&cell));
        }
    }

    tm.compute(&active_columns, true);

    {
    let seg = &tm.get_segments(expected_active_cells[0])[1];
    assert_approx_eq!(0.3, seg.synapses[0].permanence, 0.01);
    assert_approx_eq!(0.3, seg.synapses[1].permanence, 0.01);
    }

    {
    let seg = &tm.get_segments(other_bursting_cell)[0];
    assert_approx_eq!(0.3, seg.synapses[0].permanence, 0.01);
    assert_approx_eq!(0.3, seg.synapses[1].permanence, 0.01);
    }
}   

#[test]
pub fn test_no_new_segment_if_not_enough_winner_cells() {
    let mut tm = create_tm();
    tm.max_new_synapse_count = 3;
    let zero_columns = [];
    let active_columns = [0];
    
    tm.compute(&zero_columns, true);
    tm.compute(&active_columns, true);
    
    assert_eq!(0, tm.num_segments());
}

#[test]
pub fn test_new_segment_add_synapses_to_subset_of_winner_cells() {
    let mut tm = create_tm();
    tm.max_new_synapse_count = 2;
    let previous_active_columns = [0,1,2];
    let active_columns = [4];
    
    tm.compute(&previous_active_columns, true);
    
    assert_eq!(3, tm.winner_cells.len());
    
    tm.compute(&active_columns, true);
    
    assert_eq!(1, tm.winner_cells.len());

    let winner_cell = tm.winner_cells.iter().next().unwrap();

    assert_eq!(1, tm.get_segments(*winner_cell).len());
    assert_eq!(2, tm.get_segments(*winner_cell)[0].synapses.len());
    
    for synapse in &tm.get_segments(*winner_cell)[0].synapses {
        assert_approx_eq!(0.21, synapse.permanence, 0.01);
        assert_eq!(true, tm.prev_winner_cells.contains(&synapse.cell));
    }
}

#[test]
pub fn test_new_segment_add_synapses_to_all_winner_cells() {
    let mut tm = create_tm();
    tm.max_new_synapse_count = 4;
    let previous_active_columns = [0,1,2];
    let active_columns = [4];
    
    tm.compute(&previous_active_columns, true);
    
    assert_eq!(3, tm.winner_cells.len());
    
    tm.compute(&active_columns, true);
    
    assert_eq!(1, tm.winner_cells.len());

    let winner_cell = tm.winner_cells.iter().next().unwrap();

    assert_eq!(1, tm.get_segments(*winner_cell).len());
    
    let synps = &tm.get_segments(*winner_cell)[0].synapses;
    
    assert_eq!(synps.len(), tm.prev_winner_cells.len());

    for synapse in synps {
        assert_approx_eq!(0.21, synapse.permanence, 0.01);
        assert_eq!(true, tm.prev_winner_cells.contains(&synapse.cell));
    }
}

#[test]
pub fn test_matching_segment_add_synapses_to_subset_of_winner_cells() {
    let mut tm = create_tm_cells(1);
    tm.min_threshold = 1;

    let previous_active_columns = [0,1,2,3];
    let previous_winner_cells = [ tm.get_cell(0), tm.get_cell(1), tm.get_cell(2), tm.get_cell(3)];
    let active_columns = [4];
    
    {
    let mut seg = tm.create_segment(tm.get_cell(4));
    seg.create_synapse(tm.get_cell(0), 0.5);
    tm.add_segment(seg);
    }

    tm.compute(&previous_active_columns, true);

    assert_eq!(previous_winner_cells.len(), tm.winner_cells.len());
    for cell in &previous_winner_cells {
        assert_eq!(true, tm.winner_cells.contains(&cell));
    }

    tm.compute(&active_columns, true);

    let synps = &tm.get_segments(tm.get_cell(4))[0].synapses;
    
    assert_eq!(3, synps.len());

    for synapse in synps {
        if synapse.cell.index(1) == 0 {
            continue;
        }

        assert_approx_eq!(0.21, synapse.permanence, 0.01);
        assert_eq!(true, synapse.cell.index(1) == 1 || synapse.cell.index(1) == 2 || synapse.cell.index(1) == 3);
    }
}

#[test]
pub fn test_matching_segment_add_synapses_to_all_winner_cells() {
    let mut tm = create_tm_cells(1);
    tm.min_threshold = 1;

    let previous_active_columns = [0,1];
    let previous_winner_cells = [ tm.get_cell(0), tm.get_cell(1)];
    let active_columns = [4];
    
    {
    let mut seg = tm.create_segment(tm.get_cell(4));
    seg.create_synapse(tm.get_cell(0), 0.5);
    tm.add_segment(seg);
    }

    tm.compute(&previous_active_columns, true);

    assert_eq!(previous_winner_cells.len(), tm.winner_cells.len());
    for cell in &previous_winner_cells {
        assert_eq!(true, tm.winner_cells.contains(&cell));
    }

    tm.compute(&active_columns, true);

    let synps = &tm.get_segments(tm.get_cell(4))[0].synapses;
    
    assert_eq!(2, synps.len());

    for synapse in synps {
        if synapse.cell.cell == 0 {
            continue;
        }

        assert_approx_eq!(0.21, synapse.permanence, 0.01);
        assert_eq!(true, synapse.cell.cell == 1);
    }
}

 /**
    * When a segment becomes active, grow synapses to previous winner cells.
    *
    * The number of grown synapses is calculated from the "matching segment"
    * overlap, not the "active segment" overlap.
    */
#[test]
pub fn test_active_segment_grow_synapses_according_to_potential_overlap() {
    let mut tm = create_tm_cells(1);
    tm.min_threshold = 1;
    tm.activation_threshold = 2;
    tm.max_new_synapse_count = 4;

    let previous_active_columns = [0,1,2,3,4];
    let previous_winner_cells = [ tm.get_cell(0), tm.get_cell(1), tm.get_cell(2) ,tm.get_cell(3), tm.get_cell(4)];
    let active_columns = [5];
    
    {
    let mut seg = tm.create_segment(tm.get_cell(5));
    seg.create_synapse(tm.get_cell(0), 0.5);
    seg.create_synapse(tm.get_cell(1), 0.5);
    seg.create_synapse(tm.get_cell(2), 0.5);
    tm.add_segment(seg);
    }

    tm.compute(&previous_active_columns, true);

    assert_eq!(previous_winner_cells.len(), tm.winner_cells.len());
    for cell in &previous_winner_cells {
        assert_eq!(true, tm.winner_cells.contains(&cell));
    }

    tm.compute(&active_columns, true);

    let mut cells: Vec<Cell> = tm.get_segments(tm.get_cell(5))[0].synapses.iter().map(|s| s.cell).collect();
    
    assert_eq!(4, cells.len());

    cells.sort();

    assert_eq!(tm.get_cell(0), cells[0]);
    assert_eq!(tm.get_cell(1), cells[1]);
    assert_eq!(tm.get_cell(2), cells[2]);
    assert_eq!(true, cells[3] == tm.get_cell(3) || cells[3] == tm.get_cell(4));
}

#[test]
pub fn test_destroy_weak_synapse_on_wrong_prediction() {
    let mut tm = create_tm();
    tm.initial_permanence = 0.2;
    tm.max_new_synapse_count = 4;
    tm.predicted_segment_decrement = 0.02;
    
    let previous_active_columns = [0];
    let previous_active_cells = [ tm.get_cell(0), tm.get_cell(1), tm.get_cell(2) ,tm.get_cell(3)];
    let active_columns = [2];
    let expected_active_cell = tm.get_cell(5);
    
    {
    let mut seg = tm.create_segment(expected_active_cell);
    seg.create_synapse(previous_active_cells[0], 0.5);
    seg.create_synapse(previous_active_cells[1], 0.5);
    seg.create_synapse(previous_active_cells[2], 0.5);
    seg.create_synapse(previous_active_cells[3], 0.015);
    tm.add_segment(seg);
    }

    tm.compute(&previous_active_columns, true);
    tm.compute(&active_columns, true);
    
    assert_eq!(3, tm.get_segments(expected_active_cell)[0].synapses.len());
    for syn in &tm.get_segments(expected_active_cell)[0].synapses {
        assert_eq!(true, syn.cell == previous_active_cells[0] || syn.cell == previous_active_cells[1] || syn.cell == previous_active_cells[2]);
    }
}


#[test]
pub fn test_destroy_weak_synapse_on_active_reinforce() {
    let mut tm = create_tm();
    tm.initial_permanence = 0.2;
    tm.max_new_synapse_count = 4;
    tm.predicted_segment_decrement = 0.02;
    
    let previous_active_columns = [0];
    let previous_active_cells = [ tm.get_cell(0), tm.get_cell(1), tm.get_cell(2) ,tm.get_cell(3)];
    let active_columns = [2];
    let expected_active_cell = tm.get_cell(5);
    
    {
    let mut seg = tm.create_segment(expected_active_cell);
    seg.create_synapse(previous_active_cells[0], 0.5);
    seg.create_synapse(previous_active_cells[1], 0.5);
    seg.create_synapse(previous_active_cells[2], 0.5);
    seg.create_synapse(previous_active_cells[3], 0.009);
    tm.add_segment(seg);
    }

    tm.compute(&previous_active_columns, true);
    tm.compute(&active_columns, true);
    
    assert_eq!(3, tm.get_segments(expected_active_cell)[0].synapses.len());
    for syn in &tm.get_segments(expected_active_cell)[0].synapses {
        assert_eq!(true, syn.cell == previous_active_cells[0] || syn.cell == previous_active_cells[1] || syn.cell == previous_active_cells[2]);
    }
}


#[test]
pub fn test_recycle_weakest_synapse_to_make_room_for_new_synapse() {
    let mut tm = create_tm_custom(100, 1);
    tm.min_threshold = 1;
    tm.permanence_increment = 0.02;
    tm.permanence_decrement = 0.02;
    tm.max_synapses_per_segment = 3;

    let previous_active_columns = [0,1,2];
    let previous_winner_cells = [ tm.get_cell(0), tm.get_cell(1), tm.get_cell(2)];
    let active_columns = [4];
   
    
    {
    let mut seg = tm.create_segment(tm.get_cell(4));
    seg.create_synapse(tm.get_cell(81), 0.6);
    seg.create_synapse(tm.get_cell(0), 0.11);
    tm.add_segment(seg);
    }

    tm.compute(&previous_active_columns, true);

    assert_eq!(previous_winner_cells.len(), tm.winner_cells.len());
    for cell in &previous_winner_cells {
        assert_eq!(true, tm.winner_cells.contains(&cell));
    }

    tm.compute(&active_columns, true);
    
    assert_eq!(3, tm.get_segments(tm.get_cell(4))[0].synapses.len());
    for syn in &tm.get_segments(tm.get_cell(4))[0].synapses {
        assert_eq!(true, syn.cell.index(1) != 0);
    }
}

#[test]
pub fn test_recycle_least_recently_active_segment_to_make_room_for_new_segment() {
    let mut tm = create_tm_cells(1);
    tm.initial_permanence = 0.5;
    tm.permanence_increment = 0.02;
    tm.permanence_decrement = 0.02;
    tm.max_segments_per_cell = 2;

    let previous_active_columns_1 = [0,1,2];
    let previous_active_columns_2 = [3,4,5];
    let previous_active_columns_3 = [6,7,8];
    let active_columns = [9];
    let cell9 = tm.get_cell(9);
    
    tm.compute(&previous_active_columns_1, true);
    tm.compute(&active_columns, true);

    assert_eq!(1, tm.get_segments(cell9).len()); 
    tm.reset();

    tm.compute(&previous_active_columns_2, true);
    tm.compute(&active_columns, true);

    assert_eq!(2, tm.get_segments(cell9).len()); 

    let old_presynaptic: Vec<Cell> = tm.get_segments(cell9)[0].synapses.iter().map(|s| s.cell).collect();

    tm.reset();
    tm.compute(&previous_active_columns_3, true);
    tm.compute(&active_columns, true);

    assert_eq!(2, tm.get_segments(cell9).len()); 

    for segment in tm.get_segments(cell9) {
        for &cell in &old_presynaptic {
            for syn in &segment.synapses {
                assert_eq!(true, cell != syn.cell);
            }
        }
    }   
}

#[test]
pub fn test_destroy_segments_with_too_few_synapses_to_be_matching() {
    let mut tm = create_tm();
    tm.initial_permanence = 0.2;
    tm.max_new_synapse_count = 4;
    tm.predicted_segment_decrement = 0.02;

    let previous_active_columns = [0];
    let previous_active_cells = [ tm.get_cell(0), tm.get_cell(1), tm.get_cell(2), tm.get_cell(3)];
    let active_columns = [2];
    let expected_active_cell = tm.get_cell(5);
    
    {
    let mut seg = tm.create_segment(expected_active_cell);
    seg.create_synapse(previous_active_cells[0], 0.015);
    seg.create_synapse(previous_active_cells[1], 0.015);
    seg.create_synapse(previous_active_cells[2], 0.015);
    seg.create_synapse(previous_active_cells[3], 0.015);
    tm.add_segment(seg);
    }

    tm.compute(&previous_active_columns, true);
    tm.compute(&active_columns, true);
    
    assert_eq!(0, tm.get_segments(expected_active_cell)[0].synapses.len());
}


#[test]
pub fn test_punish_matching_segments_in_inactive_columns() {
    let mut tm = create_tm();
    tm.initial_permanence = 0.2;
    tm.max_new_synapse_count = 4;
    tm.predicted_segment_decrement = 0.02;

    let previous_active_columns = [0];
    let previous_active_cells = [ tm.get_cell(0), tm.get_cell(1), tm.get_cell(2), tm.get_cell(3)];
    let active_columns = [1];
    let previous_inactive_cell = tm.get_cell(81);
    
    {
    let mut seg = tm.create_segment(tm.get_cell(42));
    seg.create_synapse(previous_active_cells[0], 0.5);
    seg.create_synapse(previous_active_cells[1], 0.5);
    seg.create_synapse(previous_active_cells[2], 0.5);
    seg.create_synapse(previous_inactive_cell, 0.5);
    tm.add_segment(seg);
    }

    {
    let mut seg = tm.create_segment(tm.get_cell(43));
    seg.create_synapse(previous_active_cells[0], 0.5);
    seg.create_synapse(previous_active_cells[1], 0.5);
    seg.create_synapse(previous_inactive_cell, 0.5);
    tm.add_segment(seg);
    }

    tm.compute(&previous_active_columns, true);
    tm.compute(&active_columns, true);
    
    {
        let synapses = &tm.get_segments(tm.get_cell(42))[0].synapses;
        assert_approx_eq!(0.48, synapses[0].permanence, 0.01);
        assert_approx_eq!(0.48, synapses[1].permanence, 0.01);
        assert_approx_eq!(0.48, synapses[2].permanence, 0.01);
        assert_approx_eq!(0.50, synapses[3].permanence, 0.01);
    }

    {
        let synapses = &tm.get_segments(tm.get_cell(43))[0].synapses;
        assert_approx_eq!(0.48, synapses[0].permanence, 0.01);
        assert_approx_eq!(0.48, synapses[1].permanence, 0.01);
        assert_approx_eq!(0.50, synapses[2].permanence, 0.01);
    }
}
    
#[test]
pub fn test_add_segment_to_cell_with_fewest_segments() {
    let mut grew_on_cell1 = false;
    let mut grew_on_cell2 = false;
    
    for seed in 0..100 {
        let mut tm = create_tm();
        tm.max_new_synapse_count = 4;
        tm.predicted_segment_decrement = 0.02;
        tm.rand = UniversalRng::from_seed([seed,0,0,0]);
        
        let previous_active_columns = vec![ 1, 2, 3, 4];
        let previous_active_cells = [ tm.get_cell(4), tm.get_cell(5), tm.get_cell(6), tm.get_cell(7)];
        let active_columns = [0];

        let non_matching_cells = [ tm.get_cell(0), tm.get_cell(3) ];
        let active_cells = [ tm.get_cell(0), tm.get_cell(1), tm.get_cell(2), tm.get_cell(3) ];
        
        {
        let mut seg = tm.create_segment(non_matching_cells[0]);
        seg.create_synapse(previous_active_cells[0], 0.5);
        tm.add_segment(seg);
        }

        {
        let mut seg = tm.create_segment(non_matching_cells[1]);
        seg.create_synapse(previous_active_cells[1], 0.5);
        tm.add_segment(seg);
        }

        tm.compute(&previous_active_columns, true);
        tm.compute(&active_columns, true);
        
        assert_eq!(active_cells.len(), tm.active_cells.len());
        for cell in &active_cells {
           assert_eq!(true, tm.active_cells.contains(&cell));
        }
        
        assert_eq!(3, tm.num_segments());
        assert_eq!(1, tm.get_segments(tm.get_cell(0)).len());
        assert_eq!(1, tm.get_segments(tm.get_cell(3)).len());
        assert_eq!(1, tm.get_segments(tm.get_cell(0))[0].synapses.len());
        assert_eq!(1, tm.get_segments(tm.get_cell(3))[0].synapses.len());

        let mut segments = tm.get_segments(tm.get_cell(1)).clone();
        if segments.len() == 0 {
            let mut segments2 = tm.get_segments(tm.get_cell(2)).clone();
            assert_eq!(false, segments2.len() == 0);
            grew_on_cell2 = true;
            segments.append(&mut segments2);
        } else {
            grew_on_cell1 = true;
        }

        assert_eq!(1, segments.len());
        let synapses = &segments[0].synapses;
        assert_eq!(4, synapses.len());
        
        let mut column_check_list = previous_active_columns.clone();
        
        for synapse in synapses {
            assert_approx_eq!(0.2, synapse.permanence, 0.01);
            
            let column = synapse.cell.column as usize;
            let mut exists = false;
            let mut index = 0;
            for (i,&c) in column_check_list.iter().enumerate() {
                if c == column {
                    exists = true;
                    index = i;
                    break;
                }
            }
            assert_eq!(true, exists);
            column_check_list.swap_remove(index);
        }
        
        assert_eq!(0, column_check_list.len());
    }
    
    assert_eq!(true, grew_on_cell1);
    assert_eq!(true, grew_on_cell2);
}


