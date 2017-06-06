

use std;
use std::ops::{Index, Range};
use std::option::Option;
use std::cmp;
use algo::{PotentialPool, Topology};
use rand::{Rng, XorShiftRng, SeedableRng};
use collect_slice::CollectSlice;
use util::universal_rand::*;
use util::ClipExt;
use quickersort;
use std::collections::HashSet;
use rayon::prelude::*;

pub struct SpatialPooler {
    pub rand: UniversalRng,
    pub column_potential: PotentialPool,

    pub iteration_num: u32,
    pub iteration_learn_num: u32,

    pub inhibition_radius: usize,
    pub potential_radius: i32,
    pub potential_pct: f64,
    pub global_inhibition: bool,
    pub local_area_density: f64,
    pub num_active_columns_per_inh_area: f64,
    pub stimulus_threshold: f32,

    pub min_pct_overlap_duty_cycles: f32,
    pub min_pct_active_duty_cycles: f32,
    pub predicted_segment_decrement: f32,
    pub duty_cycle_period: u32,
    pub max_boost: f32,
    pub wrap_around: bool,

    pub num_inputs: usize, //product of input dimensions
    pub num_columns: usize, //product of column dimensions

    pub syn_perm_options: SynapsePermenenceOptions,

    pub init_connected_pct: f32,

    pub update_period: u32,

    /** Total number of columns */
    pub column_dimensions: Vec<usize>,
    /** What will comprise the Layer input. Input (i.e. from encoder) */
    pub input_dimensions: Vec<usize>,

    pub column_topology: Topology,
    pub input_topology: Topology,

    pub overlap_duty_cycles: Vec<f32>,
    pub active_duty_cycles: Vec<f32>,
    pub min_overlap_duty_cycles: Vec<f32>,
    pub min_active_duty_cycles: Vec<f32>,
    pub boost_factors: Vec<f32>,

    pub overlaps: Vec<f32>,
    pub winner_columns: Vec<usize>,
    pub tie_broken_overlaps: Vec<f32>,
    //pub kdtree: KdTree<usize>,
    pub compability_mode: bool,
}

pub struct SynapsePermenenceOptions {
    pub inactive_dec: f32,
    pub active_inc: f32,
    pub connected: f32,
    pub below_stimulus_inc: f32,
    pub min: f32,
    pub max: f32,
    pub trim_threshold: f32,
}



pub enum SPError {
    InvalidInhibitionParameters,
}

impl SpatialPooler {
    pub fn new(input_dimensions: Vec<usize>, column_dimensions: Vec<usize>) -> SpatialPooler {
        let column_size = column_dimensions.iter().product::<usize>();
        let input_size = input_dimensions.iter().product::<usize>();

        let column_topology = Topology::new(&column_dimensions);
        let input_topology = Topology::new(&input_dimensions);

        let c = SpatialPooler {
            iteration_num: 0,
            iteration_learn_num: 0,
            potential_radius: 16,
            potential_pct: 0.5,
            global_inhibition: false,
            local_area_density: -1.0,

            min_pct_overlap_duty_cycles: 0.001,
            min_pct_active_duty_cycles: 0.001,
            duty_cycle_period: 1000,
            max_boost: 10.0,
            wrap_around: true,
            num_inputs: input_size,
            num_columns: column_size,

            inhibition_radius: 0,
            update_period: 50,
            init_connected_pct: 0.5,

            syn_perm_options: SynapsePermenenceOptions {
                inactive_dec: 0.008,
                min: 0.0,
                max: 1.0,
                connected: 0.10,
                below_stimulus_inc: 0.10 / 10.0,
                active_inc: 0.05,
                trim_threshold: 0.05 / 2.0,
            },

            num_active_columns_per_inh_area: 0.0,
            predicted_segment_decrement: 0.0,
            stimulus_threshold: 0.0,

            column_dimensions: column_dimensions,
            input_dimensions: input_dimensions,

            column_topology: column_topology,
            input_topology: input_topology,

            overlap_duty_cycles: vec![0f32; column_size],
            active_duty_cycles: vec![0f32; column_size],
            min_overlap_duty_cycles: vec![0f32; column_size],
            min_active_duty_cycles: vec![0f32; column_size],
            boost_factors: vec![1f32; column_size],

            column_potential: PotentialPool::new(0, 0),
            rand: UniversalRng::from_seed([42, 0, 0, 0]),

            overlaps: vec![0.0; column_size],
            winner_columns: vec![0; column_size],
            tie_broken_overlaps: vec![0.0; column_size],
            compability_mode: false,
        };
        c
    }


    pub fn post_init(&mut self) {
        self.syn_perm_options.below_stimulus_inc = self.syn_perm_options.connected / 10.0;
        self.syn_perm_options.trim_threshold = self.syn_perm_options.active_inc / 2.0;
        if self.potential_radius == -1 {
            self.potential_radius = self.num_inputs as i32;
        }
    }

    pub fn init(&mut self) -> Option<SPError> {
        if self.num_active_columns_per_inh_area == 0.0 &&
           (self.local_area_density == 0.0 || self.local_area_density > 0.5) {
            Some(SPError::InvalidInhibitionParameters)
        } else {
            self.post_init();
            self.gen_column_potential();
            self.connect_and_configure_inputs();
            None
        }
    }

    pub fn compute(&mut self, input_vector: &[bool], learn: bool) {
        self.update_iteration_number(learn);
        self.calculate_overlaps(input_vector);
        self.boost(learn);
        self.inhibit_columns();

        if learn {
            self.adapt_synapses(input_vector);
            self.update_duty_cycles();
            self.bump_up_weak_columns();
            self.update_boost_factors();
            if self.iteration_num % self.update_period == 0 {
                self.update_inhibition_radius();
                self.update_min_duty_cycles();
            }
        }
    }

    pub fn adapt_groups(&mut self) {

    }

    pub fn adapt_synapses(&mut self, input_vector: &[bool]) {
        for column in &self.winner_columns {
            for val in self.column_potential
                    .connections_by_index_mut(*column)
                    .iter_mut() {
                if input_vector[val.index as usize] {
                    val.permanence += self.syn_perm_options.active_inc;
                } else {
                    val.permanence -= self.syn_perm_options.inactive_dec;
                }
            }
            self.column_potential
                .update_permanences(*column,
                                               true,
                                               (self.stimulus_threshold + 0.5) as i32,
                                               &self.syn_perm_options);
        }
    }

    pub fn update_duty_cycles(&mut self) {
        let period = if self.duty_cycle_period > self.iteration_num {
            self.iteration_num as f32
        } else {
            self.duty_cycle_period as f32
        };

        for (index, &val) in self.overlaps.iter().enumerate() {
            self.overlap_duty_cycles[index] = ((self.overlap_duty_cycles[index] * (period - 1.0)) +
                                               (val > 0.0) as usize as f32) /
                                              period;
        }

        for val in self.active_duty_cycles.iter_mut() {
            *val = (*val * (period - 1.0)) / period;
        }
        for &val in self.winner_columns.iter() {
            self.active_duty_cycles[val] = ((self.active_duty_cycles[val] * (period - 1.0)) + 1.0) /
                                           period;
        }

    }

    pub fn bump_up_weak_columns(&mut self) {
        let connected = self.syn_perm_options.connected;
        for ((column, overlap_duty_cycle), min_overlap_duty_cycle) in
            self.overlap_duty_cycles
                .iter()
                .enumerate()
                .zip(self.min_overlap_duty_cycles.iter()) {
            if (min_overlap_duty_cycle > overlap_duty_cycle) {
                for val in self.column_potential.connections_by_index_mut(column) {
                    val.permanence += self.syn_perm_options.below_stimulus_inc;
                }
                self.column_potential
                    .update_permanences(column,
                                                   true,
                                                   (self.stimulus_threshold + 0.5) as i32,
                                                   &self.syn_perm_options);
            }
        }
    }

    pub fn update_boost_factors(&mut self) {
        let mut got_elements = false;
        for &val in &self.min_active_duty_cycles {
            if (val > 0.0) {
                got_elements = true;
                break;
            }
        }
        if (got_elements) {
            for ((boost, &min_active), &active) in
                self.boost_factors
                    .iter_mut()
                    .zip(self.min_active_duty_cycles.iter())
                    .zip(self.active_duty_cycles.iter()) {
                *boost = if active > min_active {
                    1.0
                } else {
                    let ma = if min_active == 0.0 { 1.0 } else { min_active };
                    (((1.0 - self.max_boost) / ma) * active) + self.max_boost
                };
            }
        }
    }

    pub fn update_min_duty_cycles(&mut self) {
        if self.global_inhibition || self.inhibition_radius > self.num_inputs {
            self.update_min_duty_cycles_global();
        } else {
            self.update_min_duty_cycles_local();
        }
    }

    /**
     * Updates the minimum duty cycles in a global fashion. Sets the minimum duty
     * cycles for the overlap and activation of all columns to be a percent of the
     * maximum in the region, specified by {@link Connections#getMinOverlapDutyCycles()} and
     * minPctActiveDutyCycle respectively. Functionality it is equivalent to
     * {@link #updateMinDutyCyclesLocal(Connections)}, but this function exploits the globalness of the
     * computation to perform it in a straightforward, and more efficient manner.
     * 
     * @param c
     */
    pub fn update_min_duty_cycles_global(&mut self) {
        let m = self.min_pct_overlap_duty_cycles *
                *self.overlap_duty_cycles
                     .iter()
                     .max_by(|a, b| a.partial_cmp(b).unwrap())
                     .unwrap();
        let m2 = self.min_pct_active_duty_cycles *
                 *self.active_duty_cycles
                      .iter()
                      .max_by(|a, b| a.partial_cmp(b).unwrap())
                      .unwrap();
        for value in &mut self.min_overlap_duty_cycles {
            *value = m;
        }
        for value in &mut self.min_active_duty_cycles {
            *value = m2;
        }
    }

    /**
     * Updates the minimum duty cycles. The minimum duty cycles are determined
     * locally. Each column's minimum duty cycles are set to be a percent of the
     * maximum duty cycles in the column's neighborhood. Unlike
     * {@link #updateMinDutyCyclesGlobal(Connections)}, here the values can be 
     * quite different for different columns.
     * 
     * @param c
     */
    pub fn update_min_duty_cycles_local(&mut self) {
        let radius = self.inhibition_radius;
        let wrapping = self.wrap_around;
        for column in 0..self.num_columns {
            let neighborhood = self.column_topology.neighborhood(column, radius, wrapping);
            let mut max_active_duty = 0.0;
            let mut max_overlap_duty = 0.0;
            for (index, val) in neighborhood.enumerate() {
                let x = self.active_duty_cycles[index] - val as f32;
                let y = self.overlap_duty_cycles[index] - val as f32;
                if x > max_active_duty {
                    max_active_duty = x;
                }
                if y > max_overlap_duty {
                    max_overlap_duty = y;
                }
            }
            self.min_active_duty_cycles[column] = max_active_duty;
            self.min_overlap_duty_cycles[column] = max_overlap_duty;
        }
    }

    pub fn boost(&mut self, learn: bool) {
        if learn {
            for (overlap, boost) in self.overlaps.iter_mut().zip(self.boost_factors.iter()) {
                *overlap *= *boost;
            }
        }
    }

    pub fn calculate_overlaps(&mut self, input_vector: &[bool]) {
        let connected = self.syn_perm_options.connected;
        for column in 0..self.num_columns {
            let mut counter = 0;
            for con in self.column_potential.connected_by_index(column) {
                counter += input_vector[con.index as usize] as usize;
            }
            self.overlaps[column] = counter as f32;
        }
    }

    pub fn update_iteration_number(&mut self, learn: bool) {
        self.iteration_num += 1;
        if learn {
            self.iteration_learn_num += 1;
        }
    }

    pub fn inhibit_columns(&mut self) {
        let mut density = self.local_area_density;
        if density <= 0.0 {
            let inhibitionArea = cmp::min((2 * self.inhibition_radius + 1)
                                              .pow(self.column_dimensions.len() as u32),
                                          self.num_columns as usize);
            density = self.num_active_columns_per_inh_area / inhibitionArea as f64;
            if density > 0.5 {
                density = 0.5;
            }
        }

        //Add our fixed little bit of random noise to the scores to help break ties.
        //ArrayUtils.d_add(overlaps, c.getTieBreaker());

        if self.global_inhibition ||
           self.inhibition_radius > *self.column_dimensions.iter().max().unwrap() {
            self.inhibit_columns_global(density as f32);
        } else {
            self.inhibit_columns_local(density as f32);
        }
    }

    pub fn inhibit_columns_global(&mut self, density: f32) {
        let mut numActive = (density * self.num_columns as f32) as usize;

        self.winner_columns.clear();
        for i in 0..self.num_columns {
            self.winner_columns.push(self.num_columns - i - 1);
        }

        //TODO: use quickersort.
        let ovelaps = &self.overlaps;
        self.winner_columns[0..self.num_columns].sort_by(|i, i2| {
                                                             let f1 = ovelaps[*i];
                                                             let f2 = ovelaps[*i2];
                                                             f2.partial_cmp(&f1).unwrap()
                                                         });

        while (numActive > 0) {
            if self.overlaps[self.winner_columns[numActive]] >= self.stimulus_threshold {
                break;
            }
            numActive -= 1;
        }

        self.winner_columns.truncate(numActive);
    }

    pub fn inhibit_columns_local(&mut self, density: f32) {
        let mut max_overlaps = 1.0;
        for (i, &overlap) in self.overlaps.iter().enumerate() {
            self.tie_broken_overlaps[i] = overlap;
            if max_overlaps < overlap {
                max_overlaps = overlap;
            }
        }
        let add_to_winners = max_overlaps / 1000.0;

        self.winner_columns.clear();
        let stimulus_threshold = self.stimulus_threshold;
        let inhibition_radius = self.inhibition_radius;

        for (column, &overlaps) in self.overlaps.iter().enumerate() {
            if overlaps >= stimulus_threshold {
                let neighborhood = self.column_topology
                    .neighborhood(column, inhibition_radius, self.wrap_around);

                let mut num_bigger = 0;
                let (num_total, _) = neighborhood.size_hint();
                for n_o in neighborhood.map(|index| self.tie_broken_overlaps[index]) {
                    if n_o > overlaps {
                        num_bigger += 1;
                    }
                }

                let num_active = (0.5 + density * num_total as f32) as u32;
                if num_bigger < num_active {
                    self.winner_columns.push(column);
                    self.tie_broken_overlaps[column] += add_to_winners;
                }
            }
        }
    }


    pub fn connect_and_configure_inputs(&mut self) {
        // Initialize the set of permanence values for each column. Ensure that
        // each column is connected to enough input bits to allow it to be
        // activated.

        //let mut arr = vec![0usize; self.max_potential()];
        let mut arr = vec![0usize; self.num_inputs];

        for i in 0..self.num_columns {
            let range = self.map_potential(i, true, &mut arr);
            quickersort::sort(&mut arr[range.clone()]);
            self.column_potential
                .setup_pool(i,
                            &arr[range.clone()],
                            self.init_connected_pct,
                            &self.syn_perm_options,
                            &mut self.rand);
            self.column_potential.update_permanences(i,
                                               true,
                                               (self.stimulus_threshold + 0.5) as i32,
                                               &self.syn_perm_options);
        }
        //self.column_potential.test_connections();
        // The inhibition radius determines the size of a column's local
        // neighborhood.  A cortical column must overcome the overlap score of
        // columns in its neighborhood in order to become active. This radius is
        // updated every learning round. It grows and shrinks with the average
        // number of connected synapses per column.
        self.update_inhibition_radius();
    }

    pub fn map_potential(&mut self,
                         column_index: usize,
                         wrap_around: bool,
                         into: &mut [usize])
                         -> std::ops::Range<usize> {
        let center_input = self.map_column(column_index);
        let elements_iter =
            self.input_topology
                .neighborhood(center_input, self.potential_radius as usize, wrap_around);
        let (size, _) = elements_iter.size_hint();
        let final_size = self.potential_synapses(size);

        if self.compability_mode {
            SpatialPooler::sample_into_universal(&mut self.rand, elements_iter, final_size, into)
        } else {
            SpatialPooler::sample_into(&mut self.rand, elements_iter, final_size, into)
        }
    }

    pub fn update_inhibition_radius(&mut self) {
        if self.global_inhibition {
            self.inhibition_radius = *self.column_dimensions.iter().max().unwrap();
        } else {
            let size = self.input_dimensions.len();
            let mut max = vec![0isize; size];
            let mut min = vec![0isize; size];
            let max_dim = *self.input_dimensions.iter().max().unwrap() as isize;

            let mut total: f64 = 0.0;
            for column in 0..self.num_columns {
                let cs = self.column_potential.connections_by_index(column);
                for val in &mut max {
                    *val = -1;
                }
                for val in &mut min {
                    *val = max_dim;
                }
                for syn in cs {
                    for (index, val) in self.input_topology
                            .compute_coordinates(syn.index as usize)
                            .enumerate() {
                        if max[index] < val as isize {
                            max[index] = val as isize;
                        }
                        if min[index] > val as isize {
                            min[index] = val as isize;
                        }
                    }
                }
                total += min.iter()
                    .zip(max.iter())
                    .fold(0, |acc, (x, y)| acc + y - x + 1) as f64 /
                         size as f64;
            }
            total /= self.num_columns as f64;

            let avg: f64 = self.column_dimensions
                .iter()
                .zip(self.input_dimensions.iter())
                .map(|(&c, &i)| c as f64 / i as f64)
                .sum::<f64>() / self.column_dimensions.len() as f64;
            let radius = ((avg * total) - 1.0) / 2.0;
            self.inhibition_radius = if radius < 1.0 {
                1
            } else {
                (radius + 0.5) as usize
            };
        }
    }

    pub fn max_potential(&self) -> usize {
        self.potential_synapses(self.input_dimensions
                                    .iter()
                                    .fold(1usize, |acc, &dim| {
            acc * cmp::min(dim, (self.potential_radius * 2) as usize)
        }))
    }

    pub fn gen_column_potential(&mut self) {
        //self.column_potential = PotentialPool::new(self.num_columns, self.max_potential(), self.num_inputs);
        self.column_potential =
            PotentialPool::new(self.num_columns, self.num_inputs);
    }

    pub fn potential_synapses(&self, input_size: usize) -> usize {
        (input_size as f64 * self.potential_pct + 0.5) as usize
    }

    pub fn map_column(&self, column_index: usize) -> usize {
        self.input_topology
            .index_from_coordinates(self.column_topology
                                        .compute_coordinates(column_index)
                                        .zip(self.column_dimensions.iter())
                                        .zip(self.input_dimensions.iter())
                                        .map(|((index, &col_dim), &in_dim)| {
                                                 let new_index = ((index as f32 / col_dim as f32) *
                                                                  in_dim as f32 +
                                                                  (in_dim as f32 / col_dim as f32) *
                                                                  0.5f32) as
                                                                 usize;
                                                 cmp::max(0, cmp::min(in_dim - 1, new_index))
                                             }))
    }


    pub fn sample_into<T, I, R>(rng: &mut R,
                                iterable: I,
                                amount: usize,
                                into: &mut [T])
                                -> std::ops::Range<usize>
        where I: IntoIterator<Item = T>,
              R: Rng
    {
        let mut iter = iterable.into_iter();
        let items = iter.by_ref().take(amount).collect_slice(&mut into[..]);
        // continue unless the iterator was exhausted
        if items == amount {
            for (i, elem) in iter.enumerate() {
                let k = rng.gen_range(0, i + 1 + amount);
                if let Some(spot) = into.get_mut(k) {
                    *spot = elem;
                }
            }
        }
        0..items
    }

    pub fn sample_into_universal<T: Copy, I, R>(rng: &mut R,
                                                iterable: I,
                                                amount: usize,
                                                into: &mut [T])
                                                -> std::ops::Range<usize>
        where I: IntoIterator<Item = T>,
              R: Rng
    {
        let mut iter = iterable.into_iter();
        let items = iter.by_ref().collect_slice(&mut into[..]);
        // continue unless the iterator was exhausted
        let mut upper_bound = items;
        let final_size = if items < amount { items } else { amount };
        for _ in 0..final_size {
            let random_idx = rng.next_uv_int(upper_bound as i32) as usize;
            let tmp = into[random_idx];
            for j in random_idx..upper_bound - 1 {
                into[j] = into[j + 1];
            }
            upper_bound -= 1;
            into[upper_bound] = tmp;
        }

        //for i in 0..final_size {
        //    into[i] = into[items - i - 1];
        //}
        items - final_size..items

        /*
        TIntArrayList choiceSupply = new TIntArrayList(choices);
        int upperBound = choices.length;
        for (int i = 0; i < selectedIndices.length; i++) {
            int randomIdx = random.nextInt(upperBound);
            selectedIndices[i] = (choiceSupply.removeAt(randomIdx));
            upperBound--;
        }
        Arrays.sort(selectedIndices);
        //System.out.println("sample: " + Arrays.toString(selectedIndices));
        return selectedIndices;
        */
    }
}
