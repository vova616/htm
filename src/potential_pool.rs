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
    sizes_columns: Vec<usize>,
    size_column: usize,
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
        }
        self.sizes_columns[column.index] = size;
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

    
    pub fn connections(&self, column: &Column) -> & [Synapse]  {
         &self.permanences(column)[0..self.sizes_columns[column.index]]
    } 

    pub fn connections_mut(&mut self, column: &Column) -> &mut [Synapse]  {
         let size = self.sizes_columns[column.index];
         &mut self.permanences_mut(column)[0..size]
    }     

    #[inline]
    pub fn connection_by_global_index(&self, global_index: usize) -> &Synapse  {
        &self.synapses_columns[global_index]
    }

    #[inline]
    pub fn column_global_index_to_column(&self, global_index: usize) -> usize  {
        global_index / self.size_column
    }

}

