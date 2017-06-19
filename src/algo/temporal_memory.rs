use std;
use std::collections::HashSet;
use std::cmp;
use fnv::{FnvHashMap,FnvHashSet};
use rand::{Rng, XorShiftRng, SeedableRng};
use util::{UniversalRng, UniversalNext, PeekableWhile, PeekableWhileTrait};
use util::numext::*;
use quickersort;

pub struct TemporalMemory {
    pub active_cells: FnvHashSet<Cell>,
    pub prev_active_cells: FnvHashSet<Cell>,

    pub winner_cells: FnvHashSet<Cell>,
    pub prev_winner_cells: FnvHashSet<Cell>,

    predictive_cells: FnvHashMap<Cell, u32>,

    synapse_map: FnvHashMap<Cell, FnvHashSet<SynapseLink>>,
    segments: FnvHashMap<Cell, Vec<Segment>>,

    segments_am_helper: FnvHashMap<SegmentRef, (u32,u32)>,

    pub segments_active: std::cell::Cell<Vec<SegmentScore>>,
    pub segments_matching: std::cell::Cell<Vec<SegmentScore>>,

    empty_segments: Vec<Segment>,

    /**
     * If the number of active connected synapses on a segment
     * is at least this threshold, the segment is said to be active.
    */
    pub activation_threshold: u32, // = 13;

     /**
     * If the number of synapses active on a segment is at least this
     * threshold, it is selected as the best matching
     * cell in a bursting column.
     */
    pub min_threshold: u32, // = 10;
    /** The maximum number of synapses added to a segment during learning. */
    pub max_new_synapse_count: u32, // = 20;
    /** The maximum number of segments (distal dendrites) allowed on a cell */
    pub max_segments_per_cell: u32, // = 255;
    /** The maximum number of synapses allowed on a given segment (distal dendrite) */
    pub max_synapses_per_segment: u32, // = 255;
    /** Initial permanence of a new synapse */
    pub initial_permanence: f32, // // = 0.21;
    /**
     * If the permanence value for a synapse
     * is greater than this value, it is said
     * to be connected.
     */
    pub connected_permanence: f32, // = 0.50;
    /**
     * Amount by which permanences of synapses
     * are incremented during learning.
     */
    pub permanence_increment: f32, // = 0.10;
    /**
     * Amount by which permanences of synapses
     * are decremented during learning.
     */
    pub permanence_decrement: f32, // = 0.10;

    pub predicted_segment_decrement: f32,

    pub iteration: u64,
    

    pub cells: u32,

    pub rand: UniversalRng,
}

#[derive(Copy,Clone,PartialEq,Eq,Hash)]
pub struct Cell {
    pub column: u32,
    pub cell: u32,
}



use std::cmp::{Ord,Ordering};

impl Cell {
    pub fn index(&self, max_cells: u32) -> u64 {
       self.column as u64 * max_cells as u64 + self.cell as u64
    }
}

use std::fmt;
impl fmt::Display for Cell {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match f.precision() {
            Some(n) => write!(f, "{}", self.index(n as u32)),
            None =>  write!(f, "({}, {})", self.column, self.cell)
        }
    }
}

impl fmt::Debug for Cell {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match f.precision() {
            Some(n) => write!(f, "{}", self.index(n as u32)),
            None =>  write!(f, "({}, {})", self.column, self.cell)
        }
    }
}

impl Ord for Cell {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.column > other.column {
            Ordering::Greater
        } else if self.column < other.column {
            Ordering::Less
        } else {
            self.cell.cmp(&other.cell)
        }
    }
}

impl PartialOrd for Cell {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}



#[derive(Debug,Clone,Eq,Hash,PartialEq)]
pub struct SegmentRef {
    pub cell: Cell,
    pub segment: u32,
}

impl Ord for SegmentRef {
    fn cmp(&self, other: &Self) -> Ordering {
        let cmp = self.cell.cmp(&other.cell);
        if cmp == Ordering::Equal {
            self.segment.cmp(&other.segment)
        } else {
            cmp
        }
    }
}

impl PartialOrd for SegmentRef {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub struct SegmentScore {
    pub segment: SegmentRef,
    pub matched: u32,
}

impl fmt::Debug for SegmentScore {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match f.precision() {
            Some(n) => write!(f, "({}:{})", self.segment.cell.index(n as u32), self.matched),
            None =>  write!(f, "({}:{})", self.segment.cell, self.matched)
        }
    }
}


impl Ord for SegmentScore {
    fn cmp(&self, other: &Self) -> Ordering {
        self.segment.cmp(&other.segment)
    }
}

impl PartialOrd for SegmentScore {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for SegmentScore {
    fn eq(&self, other: &Self) -> bool {
        self.segment.eq(&other.segment)
    }
}
impl Eq for SegmentScore {}
impl std::hash::Hash for SegmentScore {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.segment.hash(state);
    }
}

#[derive(Debug,Eq,PartialEq,Clone,Copy)]
enum SynapseStatus {
    Alive = 1,
    Connected = 2,
}

#[derive(Debug)]
pub struct SynapseLink {
    pub segment: SegmentRef,
    connected: bool,
}

impl Ord for SynapseLink {
    fn cmp(&self, other: &Self) -> Ordering {
        self.segment.cmp(&other.segment)
    }
}

impl PartialEq for SynapseLink {
    fn eq(&self, other: &Self) -> bool {
        self.segment.eq(&other.segment)
    }
}
impl Eq for SynapseLink {}

impl std::hash::Hash for SynapseLink {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.segment.hash(state);
    }
}

impl PartialOrd for SynapseLink {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}


#[derive(Clone)]
pub struct Synapse {
    pub cell: Cell,
    pub permanence: f32,
}

impl fmt::Debug for Synapse {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match f.precision() {
            Some(n) => write!(f, "{}", self.cell.index(n as u32)),
            None =>  write!(f, "({}, {})", self.cell, self.permanence)
        }
    }
}

#[derive(Debug,Clone)]
pub struct Segment {
    pub cell: Cell,
    pub synapses: Vec<Synapse>,
    pub last_used: u64,
}


impl Segment {
    pub fn new(column: u32, cell: u32, iteration: u64) -> Segment {
        Segment {
            cell: Cell {
                column: column,
                cell: cell,
            },
            last_used: iteration,
            synapses: Vec::new(),
        }
    } 

    pub fn new_from_cell(cell: Cell, iteration: u64) -> Segment {
        Segment {
            cell: cell,
            last_used: iteration,
            synapses: Vec::new(),
        }
    } 

   
    fn grow_synapses<R: Rng>(&mut self, active_cells: &FnvHashSet<Cell>, initial_permanence: f32, desired: u32, rand: &mut R) -> std::ops::Range<usize> {
        let mut cells = active_cells.iter().cloned().collect::<Vec<Cell>>();
        cells.sort();

        for syn in &self.synapses {
            match cells.binary_search(&syn.cell) {
                Ok(i) => { cells.remove(i); },
                Err(_) => {},
            };
        }

        let n_actual = if (desired as usize) < cells.len() {
            desired as usize
            } else { 
            cells.len() 
        };
        
        
        for _ in 0..n_actual {
            let rand = rand.next_uv_int(cells.len() as i32) as usize;
            self.create_synapse(cells[rand], initial_permanence);
            cells.remove(rand);
        }

        let len = self.synapses.len();
        len - n_actual..len
    }

    pub fn create_synapse(&mut self, cell: Cell, permanence: f32)
    {
        self.synapses.push(Synapse{cell: cell, permanence: permanence});
    }

    fn adapt_segment(&mut self, active_cells: &FnvHashSet<Cell>, map: &mut FnvHashMap<Cell, FnvHashSet<SynapseLink>>, seg_ref: &SegmentRef, perm_inc: f32, perm_dec: f32, connected: f32) {
        let mut index = 0;
        while index < self.synapses.len() {
            let mut deleted = false;
            
            {
                let synapse = &mut self.synapses[index];
                let old_perm = synapse.permanence;
                if active_cells.contains(&synapse.cell) {
                    synapse.permanence += perm_inc;
                } else {
                    synapse.permanence -= perm_dec;
                }

                if old_perm < connected {
                    if synapse.permanence >= connected {
                         map.get_mut(&synapse.cell).unwrap().replace(SynapseLink{segment: seg_ref.clone(), connected: true});
                    }
                } else if synapse.permanence < connected {
                    map.get_mut(&synapse.cell).unwrap().replace(SynapseLink{segment: seg_ref.clone(), connected: false});
                }

                if synapse.permanence > 1.0 {
                    synapse.permanence = 1.0;
                } else if (synapse.permanence < 0.00001) {
                    map.get_mut(&synapse.cell).unwrap().remove(&SynapseLink{segment: seg_ref.clone(), connected: false});
                    deleted = true;
                }
            }

            if deleted {
                self.synapses.swap_remove(index);
            } else {
                index += 1;
            } 
        }
    }
}



impl TemporalMemory {
    pub fn new(columns: u32, cells: u32) -> TemporalMemory {
        TemporalMemory {
            cells: cells,
            active_cells: FnvHashSet::default(),
            prev_active_cells: FnvHashSet::default(),

            winner_cells: FnvHashSet::default(),
            prev_winner_cells: FnvHashSet::default(),

            predictive_cells: FnvHashMap::default(),

            segments_am_helper: FnvHashMap::default(),
            synapse_map: FnvHashMap::default(),
            segments: FnvHashMap::default(),

            iteration: 1,

            segments_active: std::cell::Cell::new(Vec::new()),
            segments_matching: std::cell::Cell::new(Vec::new()),
            empty_segments: Vec::new(),

            activation_threshold: 13,
            min_threshold: 10,
            max_new_synapse_count: 20,
            max_segments_per_cell: 255,
            max_synapses_per_segment: 255,
            initial_permanence: 0.21,
            connected_permanence: 0.5,
            permanence_increment: 0.1,
            permanence_decrement: 0.1,

            predicted_segment_decrement: 0.0,

            rand: UniversalRng::from_seed([42, 0, 0, 0]),
        }
    }

    pub fn compute(&mut self, active_columns: &[usize], learn: bool) {
        self.active_cells(active_columns, learn);
        self.activate_dendrites(true);
    }

    #[inline]
    pub fn min2<T: Ord>(v1o: Option<T>, v2o: Option<T>) -> Option<T> {
        match (v1o,v2o) {
           (Some(v1), Some(v2)) => if v1 <= v2 { Some(v1) } else { Some(v2) },
           (Some(v1), None) => Some(v1),
           (None, Some(v2)) => Some(v2),
           (None, None) => None,
        }
    }

    #[inline]
    pub fn min<T: Ord>(v1o: Option<T>, v2o: Option<T>) -> Option<T> {
        let case = match (&v1o, &v2o) {
            (&Some(ref v1), &Some(ref v2)) => if v1 <= v2 { 1 } else { 2 },
            (&Some(_), &None) => 1,
            (&None, &Some(_)) => 2,
            (&None, &None) => 3,
        };
        match case {
            1 => v1o,
            2 => v2o,
            _ => None,
        }
    }

    fn add_synapses(map: &mut FnvHashMap<Cell, FnvHashSet<SynapseLink>>, segment: &mut Segment, range: std::ops::Range<usize>, segment_ref: &SegmentRef, connected: f32, max_synapses: u32) 
    {
        let mut syns = &mut segment.synapses;
        for (index, syn) in syns[range].iter().enumerate() {
            map.entry(syn.cell).or_insert(FnvHashSet::default()).replace(SynapseLink{segment: segment_ref.clone(), connected: syn.permanence >= connected});
        }

        //TODO: maybe sort and then remove, better for efficiency when syns.len() - max_synapses > log(syns.len())
        while syns.len() > max_synapses as usize {
            let index = { 
                let (i,syn) = syns.iter().enumerate().min_by(|&(_,s), &(_,s2)| s.permanence.partial_cmp(&s2.permanence).unwrap()).unwrap();
                map.get_mut(&syn.cell).unwrap().remove(&SynapseLink{segment: segment_ref.clone(), connected: false});
                i
            };
            syns.swap_remove(index);
        }
    }

   

    pub fn reset(&mut self) {
        self.winner_cells.clear();
        self.active_cells.clear();
        self.segments_matching.get_mut().clear();
        self.segments_active.get_mut().clear();
        self.active_cells.clear();
        self.predictive_cells.clear();
    }

    pub fn activate_dendrites(&mut self, learn: bool) {
        let mut segments_active = self.segments_active.get_mut();
        let mut segments_matching = self.segments_matching.get_mut();

        segments_active.clear();
        segments_matching.clear();
        self.segments_am_helper.clear();
        self.predictive_cells.clear();

        for cell in self.active_cells.iter() {
            match self.synapse_map.get(&cell) {
                Some(vec) => {
                    for syn_link in vec {
                       let mut val = &mut self.segments_am_helper.entry(syn_link.segment.clone()).or_insert((0, 0));
                       val.0 += (syn_link.connected as u32);
                       val.1 += 1;
                    }   
                },
                None => {},
            }
        }

        for (key, val) in self.segments_am_helper.iter() {
            if val.0 >= self.activation_threshold {
                segments_active.push(SegmentScore{segment: key.clone(), matched: val.1 });
            }
            if val.1 >= self.min_threshold {
                segments_matching.push(SegmentScore{segment: key.clone(), matched: val.1 });
            }
        }   

        if learn {
            for seg in segments_active.iter() {
                self.segments.get_mut(&seg.segment.cell).unwrap()[seg.segment.segment as usize].last_used = self.iteration;
            }
            self.iteration += 1;
        }
        
        quickersort::sort(&mut segments_active);
        quickersort::sort(&mut segments_matching);
        debug!("active_cells {:?}", self.active_cells);
        debug!("winner_cells {:?}", self.winner_cells);
        debug!("active {:?}", segments_active);
        debug!("matching {:?}", segments_matching);
    }

    pub fn get_segment(&self, seg_ref: &SegmentRef) -> &Segment {
        &self.segments.get(&seg_ref.cell).unwrap()[seg_ref.segment as usize]
    }

    pub fn get_segment_mut(&mut self, seg_ref: &SegmentRef) -> &mut Segment {
        &mut self.segments.get_mut(&seg_ref.cell).unwrap()[seg_ref.segment as usize]
    }

    pub fn active_cells(&mut self,active_columns: &[usize], learn: bool)
    {
        //let (mut left, mut right) = self.segments_am[0..self.segments_active+self.segments_matching].split_at(self.segments_active);

        
        let segments_active = self.segments_active.take();
        let segments_matching = self.segments_matching.take();
        {
        let mut iter = active_columns.iter().peekable();
        let mut iter_active_segs = segments_active.iter().peekable();
        let mut iter_matching_segs = segments_matching.iter().peekable();

        use std::mem;
        mem::swap(&mut self.prev_active_cells, &mut self.active_cells);
        mem::swap(&mut self.prev_winner_cells, &mut self.winner_cells);
        self.active_cells.clear();
        self.winner_cells.clear();

        loop {
            let curr_column = match iter.peek() {
                Some(&c) => Some(*c as u32),
                None => None,
            };
            let seg_active = match iter_active_segs.peek() {
                Some(ref seg) => Some(seg.segment.cell.column),
                None => None,
            };
            let seg_matching = match iter_matching_segs.peek() {
                Some(ref seg) => Some(seg.segment.cell.column),
                None => None,
            };
            let column = TemporalMemory::min(TemporalMemory::min(curr_column, seg_active), seg_matching);
            if column.is_none() {
                break;
            }
            let column_idx = match column {
                None => break,
                Some(c) => c,
            };  
            let active_column = match curr_column {
                Some(c) => if c == column_idx { iter.next(); true } else {false},
                None => false,
            };
            let active_segment = match seg_active {
                Some(c) => if c == column_idx { true } else {false},
                None => false,
            };
            let matching_segment = match seg_matching {
                Some(c) => if c == column_idx { true } else {false},
                None => false,
            };

            debug!("loop {} {:?} {:?} {:?}", column_idx, curr_column, seg_active, seg_matching);

            let mut active_segs = iter_active_segs.take_while_peek(|x| x.segment.cell.column == column_idx);
            let mut matching_segs = iter_matching_segs.take_while_peek(|x| x.segment.cell.column == column_idx);
            
            if active_column {
                if active_segment { 
                    self.activate_predicted_column(&mut active_segs, learn);
                } else {
                     //Burst whole Column
                     for cell in 0..self.cells {
                         self.active_cells.insert(Cell{column:column_idx, cell: cell});
                     }
                     if matching_segment {  
                        self.adapt_best_segment(&mut matching_segs, learn);
                     } else {
                        self.grow_segment(column_idx);
                     }      
                }
            } else {
                if matching_segment {
                    if self.predicted_segment_decrement > 0.0 {
                        self.punish_predicted_columns(&mut matching_segs);
                    }
                }
            }
            
            if matching_segment {
                matching_segs.drain();
            }
            if active_segment {
                active_segs.drain();
            }
        }
        }

        self.segments_active.replace(segments_active);
        self.segments_matching.replace(segments_matching);
    }   

    

    fn grow_segment(&mut self, column: u32) {
        let cell = TemporalMemory::least_used_cell(column, &mut self.rand, &self.segments, self.cells);
        self.winner_cells.insert(cell);
        let n_grow_desired = cmp::min(self.max_new_synapse_count, self.prev_winner_cells.len() as u32);
        if n_grow_desired > 0 {
            let mut vec = self.segments.entry(cell).or_insert(Vec::new());
            let (seg, seg_ref) = Self::insert_segment(&mut vec, Segment::new(cell.column, cell.cell, self.iteration), &mut self.synapse_map, self.max_segments_per_cell);
            let range = seg.grow_synapses(&self.prev_winner_cells, self.initial_permanence, n_grow_desired, &mut self.rand);
            TemporalMemory::add_synapses(&mut self.synapse_map, seg, range, &seg_ref, self.connected_permanence, self.max_synapses_per_segment);
            debug!("New Segment {:?}", seg);
        }
    }

    fn adapt_best_segment<'a, I: Iterator<Item=&'a SegmentScore>>(&mut self, matching_segs: &mut I, learn: bool) {
        let best_seg = matching_segs.max_by_key(|s| s.matched).unwrap();
        self.winner_cells.insert(best_seg.segment.cell);
        let seg = &mut self.segments.get_mut(&best_seg.segment.cell).unwrap()[best_seg.segment.segment as usize];
        if learn {
            //might be better to use  &self.prev_winner_cells somehow without touching the not relevant ones
            seg.adapt_segment(&self.prev_active_cells,  &mut self.synapse_map, &best_seg.segment, self.permanence_increment, self.permanence_decrement, self.connected_permanence);
            let n_grow_desired = self.max_new_synapse_count  as i32 - best_seg.matched as i32;
            if n_grow_desired > 0 {
            let range = seg.grow_synapses(&self.prev_winner_cells, self.initial_permanence, n_grow_desired as u32, &mut self.rand);
            TemporalMemory::add_synapses(&mut self.synapse_map, seg, range, &best_seg.segment, self.connected_permanence, self.max_synapses_per_segment);
            }
            debug!("Update Matching {:?}", seg);
        }
    }

    fn punish_predicted_columns<'a, I: Iterator<Item=&'a SegmentScore>>(&mut self, matching_segs: &mut I) {
         for seg_ref in matching_segs {
            let vec = &mut self.segments.get_mut(&seg_ref.segment.cell).unwrap();
            let seg = &mut vec[seg_ref.segment.segment as usize];
            
            seg.adapt_segment(&self.prev_active_cells, &mut self.synapse_map, &seg_ref.segment, -self.predicted_segment_decrement, 0.0, self.connected_permanence);
            debug!("Punish {:?}",  seg);
            //debug!("AfterPunish {:?}", seg.synapses);
        }
    }

    fn activate_predicted_column<'a, I: Iterator<Item=&'a SegmentScore>>(&mut self, active_segs: &mut I, learn: bool) {
        for active_seg in active_segs {
            debug!("Reward {:?}", active_seg.segment);
            self.active_cells.insert(active_seg.segment.cell);
            self.winner_cells.insert(active_seg.segment.cell);
            if learn {
                let seg = &mut self.segments.get_mut(&active_seg.segment.cell).unwrap()[active_seg.segment.segment as usize];
                let active_potential = active_seg.matched;    
                seg.adapt_segment(&self.prev_active_cells, &mut self.synapse_map, &active_seg.segment, self.permanence_increment, self.permanence_decrement, self.connected_permanence);
                let n_grow_desired = self.max_new_synapse_count  as i32 - active_potential as i32;
                if n_grow_desired > 0 {
                    let range = seg.grow_synapses(&self.prev_winner_cells, self.initial_permanence, n_grow_desired as u32, &mut self.rand);
                    TemporalMemory::add_synapses(&mut self.synapse_map, seg, range, &active_seg.segment, self.connected_permanence, self.max_synapses_per_segment);
                }
            }
        }
    }

    pub fn get_cell(&self, cell: usize) -> Cell {
        Cell{ column: (cell / self.cells  as usize) as u32, cell:  (cell % self.cells as usize) as u32 }
    }

    pub fn create_segment(&self, cell: Cell) -> Segment {
        Segment::new_from_cell(cell, self.iteration)
    }

    pub fn add_segment(&mut self, mut segment: Segment) {
        let mut vec = self.segments.entry(segment.cell).or_insert(Vec::new());
        let (seg, seg_ref) = Self::insert_segment(&mut vec, segment, &mut self.synapse_map, self.max_segments_per_cell);
        let range =  0..seg.synapses.len();
        TemporalMemory::add_synapses(&mut self.synapse_map, seg, range, &seg_ref, self.connected_permanence, self.max_synapses_per_segment);
    }   

    fn insert_segment<'a>(segments: &'a mut Vec<Segment>, segment: Segment, map: &mut FnvHashMap<Cell, FnvHashSet<SynapseLink>>, max_segments: u32) -> (&'a mut Segment, SegmentRef) {
        if segments.len() >= max_segments as usize {
            let index = {
                let (index, seg) = segments.iter().enumerate().min_by_key(|&(_,seg)| seg.last_used * (seg.synapses.len() > 0) as u64).unwrap();
                let seg_ref = SegmentRef{ cell: seg.cell, segment: index as u32};
                for syn in &seg.synapses {
                    map.get_mut(&syn.cell).unwrap().remove(&SynapseLink{segment: seg_ref.clone(), connected: false});
                }
                index
            };
            let segref = SegmentRef{ cell: segment.cell, segment: index as u32 };
            segments[index] = segment;
            (&mut segments[index], segref)
        } else {
            let index_opt = match segments.iter().enumerate().find(|&(index, seg)| seg.synapses.len() == 0) {
                Some((index,_)) => Some(index),
                None => None,
            };
            match index_opt {
                Some(index) => { 
                    let segref = SegmentRef{ cell: segment.cell, segment: index as u32 };
                    segments[index] = segment; 
                    (&mut segments[index], segref)
                },
                None => {
                    let segref = SegmentRef{ cell: segment.cell, segment: segments.len() as u32 };
                    segments.push(segment); 
                    (&mut segments[segref.segment as usize], segref)
                },
            }
        }
    }

    pub fn num_segments(&self) -> usize {
        self.segments.iter().fold(0, |sum, i| sum + i.1.len())
    }
    
    pub fn get_segments(&self, cell: Cell) -> &Vec<Segment> {
        match self.segments.get(&cell) {
            Some(vec) => &vec,
            None => &self.empty_segments
        }
    }

    pub fn get_predictive_cells(&mut self) -> &FnvHashMap<Cell, u32> {
        self.predictive_cells.clear();
        
        for seg in self.segments_active.get_mut() {
            *self.predictive_cells.entry(seg.segment.cell).or_insert(0) += 1;
        }
        
        &self.predictive_cells
    }

    

    pub fn least_used_cell<R: Rng>(column: u32, rand: &mut R, segments: &FnvHashMap<Cell, Vec<Segment>>, max_cells: u32) -> Cell {
        let mut min = <usize>::max_value();
        let mut counter = 0;
        for cell in 0..max_cells {
            let size = match segments.get(&Cell{column:column, cell: cell}) {
                Some(vec) => vec.len(),
                None => 0,
            };
            if min > size { 
                counter = 1;
                min = size; 
            } else if min == size {
                counter += 1;
            }
        }

        let index = rand.next_uv_int(counter as i32);
        let mut index_counter = 0;
        for cell in 0..max_cells {
            let c = Cell{column:column, cell: cell};
            let size = match segments.get(&c) {
                Some(vec) => vec.len(),
                None => 0,
            };
            if min == size { 
                if index_counter == index {
                    return c;
                }
                index_counter += 1;
            }
        }   
        panic!("shouldn't happen");
    }

}