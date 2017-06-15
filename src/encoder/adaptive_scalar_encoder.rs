
use std::ops::Range;
use encoder::ScalarEncoder;
use std::collections::VecDeque;
use std::f64;

pub struct AdaptiveScalarEncoder {
    encoder: ScalarEncoder,
    window: VecDeque<f64>,
}   



impl AdaptiveScalarEncoder {
    pub fn new(width: usize, size: usize, minmax: Option<Range<f64>>) -> AdaptiveScalarEncoder {
        Self::new_window(width, size, minmax, 300)
    }

    pub fn new_window(width: usize, size: usize, minmax: Option<Range<f64>>, window_size: usize) -> AdaptiveScalarEncoder {
        let minmax_o = minmax.unwrap_or(0.0..0.0);
        let mut encoder = AdaptiveScalarEncoder {
            encoder: ScalarEncoder::new(width, minmax_o.start, minmax_o.end, size, false),
            window: VecDeque::with_capacity(window_size),
        };
        encoder.encoder.init();
        encoder
    }   

    pub fn encode_into(&mut self, input: f64, output: &mut [bool]) {
        if !input.is_nan() {
           self.update_minmax(input);
        }
        self.encoder.encode_into(input, output);
    }

    pub fn encode(&mut self, input: f64) -> &[bool] {
        if !input.is_nan() {
           self.update_minmax(input);
        }
        self.encoder.encode(input)
    }

    fn set_encoder_params(&mut self) {
        self.encoder.internal_range = self.encoder.max - self.encoder.min;
        self.encoder.resolution = self.encoder.internal_range / (self.encoder.size - self.encoder.width) as f64;
        self.encoder.radius = self.encoder.width as f64 * self.encoder.resolution;
        self.encoder.range = self.encoder.internal_range + self.encoder.resolution;
        self.encoder.internal_size = self.encoder.size - 2 * self.encoder.padding;
    }

    pub fn get_bucket_index(&mut self, input: f64) -> Option<usize> {
        if !input.is_nan() {
            return None;
        }
        self.update_minmax(input);
        self.encoder.get_bucket_index(input)
    }

    fn update_minmax(&mut self, input: f64) {
        if self.window.len() == self.window.capacity() {
            self.window.pop_front();
        }
        self.window.push_back(input);
        if self.encoder.min == self.encoder.max {
            self.encoder.min = input;
            self.encoder.max = input + 1.0;
            self.set_encoder_params();
        } else {
            let mut max = f64::MIN;
            let mut min = f64::MAX;
            for &x in &self.window {
                if x > max {
                    max = x;
                } 
                if x < min {
                    min = x;
                }
            }
            let mut update =  false;
            if min < self.encoder.min {
                self.encoder.min = min;
                update = true;
            }
            if max > self.encoder.max {
                self.encoder.max = max;
                update = true;
            }
            if update {
                self.set_encoder_params();
            }
        }
    }
}   