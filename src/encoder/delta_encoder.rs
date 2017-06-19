use std::ops::Range;
use encoder::AdaptiveScalarEncoder;
use std::collections::VecDeque;
use std::f64;

pub struct DeltaEncoder {
    encoder: AdaptiveScalarEncoder,
}   

impl DeltaEncoder {
    pub fn new(width: usize, size: usize, minmax: Option<Range<f64>>) -> DeltaEncoder {
        Self::new_window(width, size, minmax, 300)
    }

    pub fn new_window(width: usize, size: usize, minmax: Option<Range<f64>>, window_size: usize) -> DeltaEncoder {
        let mut encoder = DeltaEncoder {
            encoder: AdaptiveScalarEncoder::new_window(width, size, minmax, window_size),
        };
        encoder
    }   

    pub fn encode(&mut self, input: f64) -> &[bool] {
		self.encoder.encode(input)
	}

    pub fn encode_into(&mut self, input: f64, output: &mut [bool]) {
        self.encoder.encode_into(input, output);
    }

    pub fn get_bucket_index(&mut self, input: f64) -> Option<usize> {
        self.encoder.get_bucket_index(input)
    }

    pub fn size(&self) -> usize {
        self.encoder.encoder.size
    }

    pub fn get_bucket_value(&self, bucket: usize) -> f64 {
       self.encoder.encoder.get_bucket_value(bucket)
    }
}