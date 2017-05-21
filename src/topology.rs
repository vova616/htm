use std::cmp;
use numext::ModuloSignedExt;
use collect_slice::CollectSlice;
use std;

pub struct Topology {
    dimension_multiples: Vec<usize>,
    dimensions: Vec<usize>,
}

pub struct TopologyCoordinateIterator<'a> {
    dimension_multiples: &'a Vec<usize>,
    i: usize,
    index: usize,
}

impl<'a> Iterator for TopologyCoordinateIterator<'a> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        if self.i < self.dimension_multiples.len() {
            let quotient = self.index / self.dimension_multiples[self.i];
            self.index %= self.dimension_multiples[self.i];
            self.i += 1;
            Some(quotient)
        } else {
            None
        }
    }


    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.dimension_multiples.len(), Some(self.dimension_multiples.len()))
    }
}

//TODO: use compile time sized arrays when ready.
pub struct TopologyIterator<'a> {
    vec: [usize; 8],
    items: [TopologyChildIterator; 8],
    index: usize,
    topology: &'a Topology,
    total_size: usize,
    dim: usize,
}

impl<'a> TopologyIterator<'a> {
    #[inline]
    fn new(topology: &Topology, index: usize, radius: usize, wrapping: bool) -> TopologyIterator {
        let mut iter = TopologyIterator {
            vec: Default::default(),
            index: 0,
            items: Default::default(),
            topology: topology,
            total_size: 0,
            dim: topology.dimensions.len(),
        };

        if wrapping {
            topology
                .dimensions
                .iter()
                .enumerate()
                .zip(topology.compute_coordinates(index))
                .map(|((i, dim), coord)| {
                    let min = coord as isize - radius as isize;
                    let max = cmp::min((coord as isize - radius as isize) + *dim as isize - 1,
                                       (coord + radius) as isize) as
                              isize + 1;
                    TopologyChildIterator {
                        upper: max,
                        lower: min,
                        index: min,
                        len: min + max - min,
                        dim: *dim as isize,
                    }
                })
                .collect_slice(&mut iter.items)
        } else {
            topology
                .dimensions
                .iter()
                .enumerate()
                .zip(topology.compute_coordinates(index))
                .map(|((i, dim), coord)| {
                    let min = cmp::max(coord as isize - radius as isize, 0) as isize;
                    let max = cmp::min((coord + radius), dim - 1) as isize + 1;
                    TopologyChildIterator {
                        upper: max,
                        lower: min,
                        index: min,
                        len: min + max - min,
                        dim: *dim as isize,
                    }
                })
                .collect_slice(&mut iter.items)
        };
        iter.total_size = iter.items[0..iter.dim]
            .iter()
            .fold(1usize, |acc, tp| acc * (tp.upper - tp.lower) as usize);


        iter
    }
}


impl<'a> Iterator for TopologyIterator<'a> {
    type Item = usize;

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.total_size, Some(self.total_size))
    }

    #[inline]
    fn next(&mut self) -> Option<usize> {
        loop {
            while self.index < self.dim - 1 {
                match self.items[self.index].next() {
                    Some(number) => {
                        self.vec[self.index] = number;
                        self.index += 1;
                    }
                    None => {
                        self.items[self.index].reset();
                        if self.index == 0 {
                            return None;
                        } else {
                            self.index -= 1;
                        }
                    }
                }
            }
            match self.items[self.index].next() {
                Some(number) => {
                    self.vec[self.index] = number;
                    let r = Some(self.topology
                                     .index_from_coordinates_slice(&self.vec[0..self.dim]));
                    return r;
                }
                None => {
                    self.items[self.index].reset();
                    if self.index == 0 {
                        return None;
                    } else {
                        self.index -= 1;
                    }
                }
            }
        }
    }
}


pub struct TopologyChildIterator {
    upper: isize,
    lower: isize,
    index: isize,
    len: isize,
    dim: isize,
}

impl Default for TopologyChildIterator {
    #[inline]
    fn default() -> TopologyChildIterator {
        TopologyChildIterator {
            upper: 0,
            lower: 0,
            index: 0,
            len: 0,
            dim: 0,
        }
    }
}

impl TopologyChildIterator {
    #[inline]
    fn reset(&mut self) {
        self.index = self.lower;
    }
}

impl Iterator for TopologyChildIterator {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        if self.index < self.len {
            let num = self.index.modulo(self.dim);
            self.index += 1;
            Some(num as usize)
        } else {
            None
        }
    }
}




impl Topology {
    pub fn new(dimensions: &Vec<usize>) -> Topology {
        Topology {
            dimension_multiples: Topology::init_dimension_multiples(dimensions),
            dimensions: dimensions.clone(),
        }
    }

    fn init_dimension_multiples(dimensions: &[usize]) -> Vec<usize> {
        let mut dimension_multiples = vec![0usize;dimensions.len()];
        let mut holder = 1;
        let len = dimensions.len();
        dimension_multiples[len - 1] = 1;
        for i in 1..len {
            holder *= dimensions[len - i] as usize;
            dimension_multiples[len - 1 - i] = holder;
        }
        dimension_multiples
    }



    pub fn neighborhood(&self,
                        center_index: usize,
                        radius: usize,
                        wrapping: bool)
                        -> TopologyIterator {
        TopologyIterator::new(self, center_index, radius, wrapping)
    }

    pub fn distance(&self, index: usize, index2: usize) -> usize {
        let mut max = 0isize;
        for (x, y) in self.compute_coordinates(index)
                .zip(self.compute_coordinates(index2)) {
            let abs = (x as isize - y as isize).abs();
            if abs > max {
                max = abs;
            }
        }
        max as usize
    }

    pub fn neighborhood_size(&self, index: usize, radius: usize, wrapping: bool) -> usize {
        self.neighborhood(index, radius, wrapping).total_size
    }

    #[inline]
    pub fn compute_coordinates(&self, index: usize) -> TopologyCoordinateIterator {
        TopologyCoordinateIterator {
            dimension_multiples: &self.dimension_multiples,
            i: 0,
            index: index,
        }
    }

    #[inline]
    pub fn index_from_coordinates_slice(&self, coordinates: &[usize]) -> usize {
        coordinates
            .iter()
            .zip(self.dimension_multiples.iter())
            .fold(0usize, |acc, (x, y)| acc + (x * y))
    }

    #[inline]
    pub fn index_from_coordinates<I: Iterator<Item = usize>>(&self, coordinates: I) -> usize {
        coordinates
            .zip(self.dimension_multiples.iter())
            .fold(0usize, |acc, (x, y)| acc + (x * y))
    }
}
