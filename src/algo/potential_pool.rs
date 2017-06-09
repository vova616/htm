use std;
use algo::{SpatialPooler, SynapsePermenenceOptions};
use rand::Rng;
use util::{ClipExt,DynamicContainer};
use util::universal_rand::*;
use quickersort;

#[derive(Debug)]
pub struct PotentialPool {
    synapses: DynamicContainer<Synapse>,
    connected_len: Vec<usize>,
}

#[derive(Debug,Clone)]
pub struct Synapse {
    pub index: usize,
    pub permanence: f32,
}

impl Default for Synapse {
    fn default() -> Self {
        Synapse {
            index: <usize>::max_value(),
            permanence: std::f32::INFINITY,
        }   
    }
}

impl PotentialPool {
    pub fn new(column_size: usize, max_potential: usize) -> PotentialPool {
        PotentialPool {
            synapses: DynamicContainer::new(column_size, max_potential),
            connected_len: vec![0; column_size],
        }
    }

    pub fn setup_pool<R: Rng>(&mut self,
                              index: usize,
                              potential: &[usize],
                              init_connected_pct: f32,
                              options: &SynapsePermenenceOptions,
                              rand: &mut R) {
        let connected_pct = init_connected_pct as f32;

        for &value in potential {
            let perm = if rand.next_f32() <= connected_pct {
                options.connected + (options.max - options.connected) * rand.next_f32()
            } else {
                options.connected * rand.next_f32()
            };

            let syn = if perm > options.trim_threshold {
                Synapse {
                    index: value ,
                    permanence: ((perm * 100000.0) as i32 as f32 / 100000.0),
                }
            } else {
                Synapse {
                    index: value ,
                    permanence: 0.0,
                }
            };
            self.synapses.insert(index, syn);
        }
        self.sort_input_synapses(index, options.connected);
    }

    pub fn sort_input_synapses(&mut self, index: usize, connected: f32) {
        let count = self.synapses.sort_pivot_children(index, |syn| syn.permanence >= connected);
        self.connected_len[index] = count;
    }

    pub fn update_permanences(&mut self,
                                         index: usize,
                                         raise_prem: bool,
                                         stimulus_threshold: i32,
                                         options: &SynapsePermenenceOptions) {
        if raise_prem {
            self.raise_permanence_to_threshold(index, stimulus_threshold, options);
        }
        for mut value in self.connections_by_column_mut(index) {
            if value.index >= 0 {
                if value.permanence <= options.trim_threshold {
                    value.permanence = 0.0;
                } else {
                    value.permanence = value.permanence.clip(options.min, options.max);
                }
            }
        }
        self.sort_input_synapses(index, options.connected);
    }

    pub fn raise_permanence_to_threshold(&mut self,
                                         index: usize,
                                         stimulus_threshold: i32,
                                         options: &SynapsePermenenceOptions) {
        let perms = self.connections_by_column_mut(index);
        loop {
            let num_connected =
                perms
                    .iter()
                    .fold(0,
                          |acc, val| acc + (val.permanence >= options.connected) as i32);
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



    pub fn connected_by_column(&self, index: usize) -> &[Synapse] {
        self.synapses.children_sized(index, self.connected_len[index])
    }

    pub fn connections_by_column(&self, index: usize) -> &[Synapse] {
        self.synapses.children(index)
    }

    pub fn connections_by_column_mut(&mut self, index: usize) -> &mut [Synapse] {
         self.synapses.children_mut(index)
    }
}
