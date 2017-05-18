use column::Column;
use std;
use bit_vec::BitVec;
use spatial_pooler::{SpatialPooler,SynapsePermenenceOptions};
use rand::Rng;
use numext::ClipExt;
use universal_rand::*;
use quickersort;

#[derive(Debug)]
pub struct PotentialPool {
    pub synapses_columns: Vec<Synapse>,
    synapses_input: Vec<usize>,
    sizes_columns: Vec<usize>,
    sizes_input: Vec<(usize,usize)>,
    size_column: usize,
    size_input: usize
}

#[derive(Debug,Clone)]
pub struct Synapse {
    pub index: isize,
    pub permanence: f32,
}

impl PotentialPool {
    pub fn new(column_size: usize, max_potential: usize, input_size: usize) -> PotentialPool {
        PotentialPool {
            synapses_columns: vec![Synapse{index:-2,permanence:0.0}; column_size * max_potential],
            sizes_columns: vec![0; column_size],
            size_column: max_potential,

            synapses_input: vec![0; 0],
            sizes_input: vec![(0,0); input_size],
            size_input: 0,
        }
    }

    pub fn setup_pool<R: Rng>(&mut self, column: &Column, potential: &[usize], init_connected_pct: f32, options: &SynapsePermenenceOptions, rand: &mut R ) {
        let mut size = 0;
        {
            let connected_pct = init_connected_pct as f32;
            let perms = self.permanences_mut(column);
            
            for &value in potential {
                let perm = if  rand.next_f32() <= connected_pct {
                    options.connected + (options.max - options.connected) * rand.next_f32()
                } else {
                    options.connected * rand.next_f32()
                };
                
                if perm > options.trim_threshold {
                    perms[size] = Synapse{index:value as isize, permanence: ((perm * 100000.0) as i32 as f32 / 100000.0) };
                } else {
                    perms[size] = Synapse{index:value as isize, permanence: 0.0};
                }
                size += 1;
            }
            //perms[0..size].sort_by_key(|syn| syn.index);
            //for x in perms[0..size].iter() {
            //    println!("{:?}", x.permanence);
            //}
        }
        self.sizes_columns[column.index] = size;
        
    }

    pub fn setup_input_pool(&mut self, options: &SynapsePermenenceOptions)
    {
        for i in 0..self.sizes_columns.len() {
            let start_index = i * self.size_column;
            let end_index = start_index + self.sizes_columns[i];
            for syn in &self.synapses_columns[start_index .. end_index] {
                self.sizes_input[syn.index as usize].0 += 1;
            }
        }
        let max = (*self.sizes_input.iter().max_by_key(|&t| t.0).unwrap()).0;
        self.size_input = max;
        self.synapses_input = vec![0; self.sizes_input.len() * self.size_input];

        for val in &mut self.sizes_input {
            (*val).0 = 0;
        }

        for i in 0..self.sizes_columns.len() {
            let start_index = i * self.size_column;
            let end_index = start_index + self.sizes_columns[i];
            let mut index = start_index;
            for syn in &self.synapses_columns[start_index .. end_index] {
                let sizes = &mut self.sizes_input[syn.index as usize];
                self.synapses_input[syn.index as usize * max + sizes.0] = index;
                (*sizes).0 += 1;
                index += 1;
            }
        }

        for input in 0..self.sizes_input.len() {
            self.sort_input_synapses(input, options.connected);
        }
    }

    pub fn update_input_synapses<'a, I: Iterator<Item=&'a usize>>(&mut self, input_indecis: I, connected: f32)
    {
        for input in input_indecis {
            self.sort_input_synapses(*input, connected);
        }
    }       

    pub fn sort_input_synapses(&mut self, index: usize, connected: f32) {
        
        let range = self.connections_by_input_range(index, false);
        if (range.end - range.start <= 0) {
            return;
        }
        let arr = &self.synapses_columns;

        {
            //Faster sort with pivot connected O(n) 
            //Maybe a sort with pivot and then index is slighty better for cache
            let mut pivot = range.start;
            for i in range.clone() {
                let val = self.synapses_input[i];
                if arr[val].permanence >= connected {
                    if (pivot != i) {
                        self.synapses_input[i] = self.synapses_input[pivot];
                        self.synapses_input[pivot] = val;
                    }
                    pivot += 1;
                }
            }
            self.sizes_input[index].1 = pivot - range.start;
        }
    }

    pub fn update_permanences_for_column(&mut self, column_index: usize, raise_prem: bool, stimulus_threshold: i32, options: &SynapsePermenenceOptions)
    {   
        if raise_prem {
            self.raise_permanence_to_threshold(column_index, stimulus_threshold, options);
        }
        let perms = self.connections_by_index_mut(column_index);
        for mut value in &mut *perms {
            if value.index >= 0 {
                if (value.permanence <= options.trim_threshold) {
                    value.permanence = 0.0;
                } else {
                    value.permanence = value.permanence.clip(options.min, options.max);
                }
            }
        }
    }

    pub fn raise_permanence_to_threshold(&mut self, column_index: usize, stimulus_threshold: i32, options: &SynapsePermenenceOptions) {
        let perms = self.connections_by_index_mut(column_index);
        loop {
            let num_connected = perms.iter().fold(0, |acc, val| acc + (val.permanence > options.connected) as i32);
            if num_connected >= stimulus_threshold {
                break;
            }
            for value in &mut *perms {
                if value.index >= 0 {
                    value.permanence += options.below_stimulus_inc;
                }
            }
        }    
    }

    
    pub fn find_index(arr: &[Synapse], index: isize) -> usize {
        arr.binary_search_by(|syn| syn.index.cmp(&index)).unwrap()
    }

    pub fn permanences_mut(&mut self, column: &Column) -> &mut [Synapse] {
        let index = column.index * self.size_column;
        let end_index = index + self.size_column;
        &mut self.synapses_columns[index..end_index]
    }

    pub fn permanences(&self, column: &Column) -> & [Synapse] {
        let index = column.index * self.size_column;
        let end_index = index + self.size_column;
        &self.synapses_columns[index..end_index]
    }   

    pub fn permanences_by_index_mut(&mut self, index: usize) -> &mut [Synapse] {
        let start_index = index * self.size_column;
        let end_index = start_index + self.size_column;
        &mut self.synapses_columns[start_index..end_index]
    }

    pub fn permanences_by_index(&self, index: usize) -> & [Synapse] {
        let start_index = index * self.size_column;
        let end_index = start_index + self.size_column;
        &self.synapses_columns[start_index..end_index]
    }   

    pub fn connections_by_index(&self, index: usize) -> & [Synapse]  {
         &self.permanences_by_index(index)[0..self.sizes_columns[index]]
    } 

    pub fn connections_by_index_mut(&mut self, index: usize) -> &mut [Synapse]  {
         let size = self.sizes_columns[index];
         &mut self.permanences_by_index_mut(index)[0..size]
    }   

    pub fn test_connections(&self)
    {
        println!("Testing connections");
        for column in 0..self.sizes_columns.len() {
            for (index, syn) in self.connections_by_index(column).iter().enumerate() {
                if syn.index < 0 {
                    println!("Broken index {} {}", column, index);
                    break;
                }
            }
        }
    }

    pub fn connections_by_input(&self, input_index: usize, only_connected: bool) -> &[usize]  {
         &self.synapses_input[self.connections_by_input_range(input_index,only_connected)]
    } 

    pub fn connections_by_input_mut(&mut self, input_index: usize, only_connected: bool) -> &mut [usize]  {
         let range = self.connections_by_input_range(input_index,only_connected).clone();
         &mut self.synapses_input[range]
    } 

    pub fn connections_by_input_range(&self, input_index: usize, only_connected: bool) -> std::ops::Range<usize>  {
         let start_index = input_index * self.size_input;
         let end_index = if only_connected {
             start_index + self.sizes_input[input_index].1
         } else {
             start_index + self.sizes_input[input_index].0
         };
         start_index..end_index
    } 

    pub fn connected_by_input(&self, input_index: usize) -> usize  {
         self.sizes_input[input_index].1
    } 

    pub fn connections(&self, column: &Column) -> & [Synapse]  {
         &self.permanences(column)[0..self.sizes_columns[column.index]]
    } 

    pub fn connections_mut(&mut self, column: &Column) -> &mut [Synapse]  {
         let size = self.sizes_columns[column.index];
         &mut self.permanences_mut(column)[0..size]
    }     

    pub fn connection_by_global_index(&self, global_index: usize) -> &Synapse  {
        &self.synapses_columns[global_index]
    }

    #[inline]
    pub fn column_global_index_to_column(&self, global_index: usize) -> usize  {
        global_index / self.size_column
    }

}

