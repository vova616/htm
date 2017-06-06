use util::numext::*;
use std::cmp::PartialOrd;
use std::ops::{Sub, Add, Mul, Div, Range};


pub struct ScalarEncoder {
    size: usize,
    internal_size: usize,
    width: usize,
    half_width: usize,
    padding: usize,
    radius: f64,
    resolution: f64,
    wrap: bool,
    clip: bool,

    output: Vec<bool>,

    min: f64,
    max: f64,
    internal_range: f64,
    range: f64,
    
}   



impl ScalarEncoder {

    pub fn new(width: usize, min: f64, max: f64, size: usize, wrap: bool) -> ScalarEncoder {
        ScalarEncoder::new_intenal(size, width, 0.0, 0.0, min, max, wrap)
    }

    pub fn new_with_resolution(width: usize, min: f64, max: f64, resolution: f64, wrap: bool) -> ScalarEncoder {
        ScalarEncoder::new_intenal(0, width, 0.0, resolution, min, max, wrap)
    }

    pub fn new_with_radius(width: usize, min: f64, max: f64, radius: f64, wrap: bool) -> ScalarEncoder {
        ScalarEncoder::new_intenal(0, width, radius, 0.0, min, max, wrap)
    }

    fn new_intenal(size: usize, width: usize, radius: f64, resolution: f64, min: f64, max: f64, wrap: bool) -> ScalarEncoder {
        if width % 2 == 0 {
            panic!("width must be an odd number (to eliminate centering difficulty)");
        }
        if min.is_nan() || max.is_nan() {
            panic!("min or max are NaN");
        }
        if min >= max {
             panic!("maxVal must be > minVal");
        }
        let half_width = (width-1) / 2;
        let mut encoder = ScalarEncoder {
            size: size,
            internal_size: size,
            width: width,
            half_width: half_width,
            radius: radius,
            resolution: resolution,
            wrap: wrap,
            clip: false,
            padding: if wrap { 0 } else { half_width },
            output: Vec::new(),
            min: min,
            max: max,
            range: max - min,
            internal_range: max - min,
        };

        encoder.init();

        encoder.internal_size = encoder.size - 2 * encoder.padding;

        encoder
    }

 
    fn init(&mut self) {
        if self.size != 0 {
            self.resolution = if !self.wrap {
                self.internal_range / (self.size - self.width) as f64
            } else {
                self.internal_range / self.size as f64
            };
            self.radius = self.width as f64 * self.resolution;
            self.range = if self.wrap  {
                self.internal_range
            }else{
                self.internal_range + self.resolution
            };
        } else {
            if self.radius != 0.0  {
                self.resolution = self.radius / self.width as f64;
            } else if self.resolution != 0.0 {
                self.radius = self.resolution * self.width as f64;
            } else {
                panic!("One of n, radius, resolution must be specified for a ScalarEncoder");
            }

            self.range = if self.wrap {
                self.internal_range
            } else {
                self.internal_range + self.resolution
            };

            let n = self.width as f64 * self.range / self.radius + 2.0 * self.padding as f64;
            self.size = (n + 0.5) as usize;
        }
    }

    fn get_encode_range(&self, input: f64) -> Range<isize> {
        let bucket = self.get_first_on_bit(input).unwrap();
        let mut minbin = bucket;
        let mut maxbin = bucket + 2 * self.half_width as isize;
        minbin..maxbin
    }

    fn encode_into_internal(input: f64, output: &mut [bool], mut range: Range<isize>, size: isize, wrap: bool) {
        for v in &mut output[..] {
            *v = false;
        }

        if input.is_nan_generic() {
            return;
        }   
        
        if wrap {
            if range.end >= size {
                for v in &mut output[0..(range.end - size + 1) as usize] {
                    *v = true;
                }
                range.end = size - 1;
            }
            if range.start < 0  {
                for v in &mut output[(size + range.start) as usize..size as usize] {
                    *v = true;
                }
                range.start = 0;
            }
        }

        for v in &mut output[range.start as usize..(range.end + 1) as usize] {
            *v = true;
        }
    }

    pub fn encode_into(&self, input: f64, output: &mut [bool]) {
       let mut range = self.get_encode_range(input);
       Self::encode_into_internal(input, output, range, self.size as isize, self.wrap);
    }

    pub fn encode(&mut self, input: f64) -> &[bool] {
        if self.output.len() <= 0 {
            self.output = vec![false; self.size];
        }
        let mut range = self.get_encode_range(input);
        Self::encode_into_internal(input, &mut self.output, range, self.size as isize, self.wrap);
        &self.output    
    }
    
    pub fn get_bucket_index(&self, input: f64) -> Option<usize> {
        match self.get_first_on_bit(input) {
            Some(minbin) => {
                //For periodic encoders, the bucket index is the index of the center bit
                Some(if self.wrap {
                    let bucketIdx = minbin + self.half_width as isize;
                    if bucketIdx < 0 {
                        (bucketIdx + self.size as isize) as usize
                    } else {
                        bucketIdx as usize
                    }
                } else {//for non-periodic encoders, the bucket index is the index of the left bit
                    minbin as usize
                })
            },
            None => {
                None
            }
        }
    }

    fn get_first_on_bit(&self, inputx: f64) -> Option<isize> {
        if inputx.is_nan() {
            return None;
        } 
        let mut input = inputx;

        /* Java code
        if input < self.min {
            if self.clip && !self.wrap {
                input = self.min;
            }else{
                panic!("input ({input} less than range ({self.min} - {self.max})");
            }
        }

        if self.wrap {
            if input >= self.max {
                 panic!("input ({input} greater than range ({self.min} - {self.max})");
            }
        } else if input > self.max {
            if self.clip {
                input = self.max;
            } else{
                 panic!("input ({input} greater than range ({self.min} - {self.max})");
            }
        }*/

        if !self.wrap {
            input = input.clip(self.min,self.max);
        } else {
            input = input.modulo(self.internal_range) + self.min;
        }

        let centerbin = if self.wrap {
            ((input - self.min) * self.internal_size as f64 / self.range) as isize
        } else {
            (((input - self.min) + self.resolution / 2.0) / self.resolution) as isize
        };

        Some(centerbin + self.padding as isize - self.half_width as isize)
    }
}