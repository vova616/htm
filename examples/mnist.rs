extern crate htm;
extern crate byteorder;

use std::fs::File;
use std::io::prelude::*;
use std::path::Path;
use std::io::BufReader;
use byteorder::{ReadBytesExt, BigEndian};
use htm::{SDRClassifier,SpatialPooler};

fn main() {
    let mut images = ImageIter::new("../train-images.idx3-ubyte");
    let mut labels = LabelIter::new("../train-labels.idx1-ubyte");

    let mut test_images = ImageIter::new("../t10k-images.idx3-ubyte");
    let mut test_labels = LabelIter::new("../t10k-labels.idx1-ubyte");


    let mut sp = SpatialPooler::new(vec![28*28], vec![64,64]);
    sp.potential_radius = 28*3;
    sp.global_inhibition = true;
    sp.num_active_columns_per_inh_area = 0.2 * sp.num_columns as f64;
    sp.syn_perm_options.active_inc = 0.00; //0.01
    sp.syn_perm_options.inactive_dec = 0.000; //0.008
    sp.syn_perm_options.trim_threshold = 0.005;
    sp.stimulus_threshold = 1.0;
    sp.syn_perm_options.connected = 0.2;
    sp.potential_pct = 20.0 / sp.potential_radius as f64;
    sp.compability_mode = true;
    sp.init();

    let mut classifier: SDRClassifier<u8> = SDRClassifier::new(vec![0], 0.1, 0.3, sp.num_columns);

    let mut input = vec![false; sp.num_inputs];

    println!("Training on:{}", images.size);

    let mut record = 0;
    for _ in 0..images.size {

        let image = images.next().unwrap();
        let label = labels.next().unwrap();

        for (inp, in_img) in input.iter_mut().zip(image.iter()) {
            *inp = *in_img > 127;
        }

        sp.compute(&input, true);

        let r = classifier.compute(record as u32,
                                   label as usize,
                                   label as u8,
                                   &sp.winner_columns[..],
                                   true,
                                   true);
        record += 1;    
    }

    println!("Testing on: {}", test_images.size);

    let mut good = 0;   
    let total = test_images.size;
    for _ in 0..test_images.size {

        let image = test_images.next().unwrap();
        let label = test_labels.next().unwrap();

        for (inp, in_img) in input.iter_mut().zip(image.iter()) {
            *inp = *in_img > 127;
        }

        sp.compute(&input, false);
        //println!("{:?}", &sp.winner_columns[..]);
        let r = classifier.compute(record as u32,
                                   label as usize,
                                   label as u8,
                                   &sp.winner_columns[..],
                                   false,
                                   true);
        for &(ref step, ref probabilities) in &r {
            let (answer, score) = probabilities
                .iter()
                .enumerate()
                .max_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap())
                .unwrap();
            if label == answer as u8 {
                good += 1;
            }
        }

        record += 1;
    }

    println!("Accuracy: {}, Total: {} Good: {}",
             good as f32 / total as f32,
             total,
             good);
}

struct ImageIter {
    buffer: Vec<u8>,
    size: usize,
    index: usize,
    width: usize,
    height: usize,
    reader: BufReader<std::fs::File>,
}

impl ImageIter {
    pub fn new<P: AsRef<Path>>(path: P) -> ImageIter {
        let ifile = File::open(path);
        if !ifile.is_ok() {
            panic!("Cannot open images file {:?}", ifile.err());
        }
        let file = ifile.unwrap();
        let mut buf_reader = BufReader::new(file);

        let header = buf_reader.read_i32::<BigEndian>().unwrap();
        if header != 2051 {
            panic!("wrong header {}", header);
        }
        let num_images = buf_reader.read_i32::<BigEndian>().unwrap();
        let width = buf_reader.read_i32::<BigEndian>().unwrap() as usize;
        let height = buf_reader.read_i32::<BigEndian>().unwrap() as usize;
        ImageIter {
            buffer: vec![0u8; width*height],
            reader: buf_reader,
            width: width,
            height: height,
            size: num_images as usize,
            index: 0,
        }
    }

    pub fn next(&mut self) -> Option<&Vec<u8>> {
        if self.index < self.size {
            let err = self.reader.read_exact(&mut self.buffer);
            if !err.is_ok() {
                panic!("counld not read pic num: {}", self.index);
            }
            //Some(ImageBuffer::from_raw(self.width as u32, self.height as u32, self.buffer).unwrap())
            Some(&self.buffer)
        } else {
            None
        }
    }
}

struct LabelIter {
    size: usize,
    index: usize,
    reader: BufReader<std::fs::File>,
}

impl Iterator for LabelIter {
    type Item = u8;

    #[inline]
    fn next(&mut self) -> Option<u8> {
        if self.index < self.size {
            let mut buffer = [0u8];
            let read = self.reader.read_exact(&mut buffer[..]);
            if !read.is_ok() {
                panic!("read error {}", self.index);
            }
            Some(buffer[0])
        } else {
            None
        }
    }
}

impl LabelIter {
    pub fn new<P: AsRef<Path>>(path: P) -> LabelIter {
        let ifile = File::open(path);
        if !ifile.is_ok() {
            panic!("Cannot open label file {:?}", ifile.err());
        }
        let file = ifile.unwrap();
        let mut buf_reader = BufReader::new(file);

        let header = buf_reader.read_i32::<BigEndian>().unwrap();
        if header != 2049 {
            panic!("wrong header {}", header);
        }
        let num_images = buf_reader.read_i32::<BigEndian>().unwrap();
        LabelIter {
            reader: buf_reader,
            size: num_images as usize,
            index: 0,
        }
    }
}
