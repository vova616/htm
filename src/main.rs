#![feature(conservative_impl_trait)]
#![allow(dead_code)]

mod spatial_pooler;
mod column;
mod topology;
mod numext;
mod potential_pool;
mod universal_rand;
mod sdr_classifier;

extern crate bit_vec;
extern crate rand;
extern crate collect_slice;
extern crate quickersort;
extern crate rayon;
extern crate time;
//extern crate image;

use rand::*;
use spatial_pooler::SpatialPooler;
use universal_rand::*;
use time::PreciseTime;


use std::fs::File;
use std::io::prelude::*;
use std::path::Path;
use std::io::BufReader;
extern crate byteorder;
use byteorder::{ReadBytesExt, WriteBytesExt, BigEndian, LittleEndian};
//use image::{ImageBuffer, Luma};
use sdr_classifier::SDRClassifier;

fn main4() {
    let mut sp = SpatialPooler::new(vec![32, 32], vec![64, 64]);
    sp.potential_radius = sp.num_inputs as i32;
    sp.global_inhibition = true;
    sp.num_active_columns_per_inh_area = 0.02 * sp.num_columns as f64;
    sp.syn_perm_options.active_inc = 0.01;
    sp.syn_perm_options.trim_threshold = 0.005;
    sp.compability_mode = true;

    {
        print!("Initializing");
        let start = PreciseTime::now();
        sp.init();
        println!(": {:?}", start.to(PreciseTime::now()));
    }

    let mut rnd = UniversalRng::from_seed([42, 0, 0, 0]);
    let mut input = vec![false; sp.num_inputs];

    let mut record = 0;
    for i in 0..10 {
        for val in &mut input {
            *val = rnd.next_uv_int(2) == 1;
        }

        sp.compute(&input, true);
        //println!(": {:?}", start.to(PreciseTime::now()).num_microseconds().unwrap() as f64 / 1000.0);
        //println!("{:?}", sp.overlaps);
        sp.winner_columns.sort();
        println!("{:?}", sp.winner_columns);
    }

}

fn main3() {
    let mut sp = SpatialPooler::new(vec![10], vec![100]);
    sp.potential_radius = 3;
    sp.global_inhibition = true;
    sp.num_active_columns_per_inh_area = 0.02 * sp.num_columns as f64;
    sp.syn_perm_options.active_inc = 0.01;
    sp.syn_perm_options.trim_threshold = 0.005;
    sp.compability_mode = true;

    let mut classifier: SDRClassifier<u8> =
        SDRClassifier::new(vec![0, 1], 0.1, 0.3, sp.num_columns);

    {
        print!("Initializing");
        let start = PreciseTime::now();
        sp.init();
        println!(": {:?}", start.to(PreciseTime::now()));
    }

    let mut rnd = UniversalRng::from_seed([42, 0, 0, 0]);
    let mut input = vec![false; sp.num_inputs];

    let mut record = 0;
    for i in 0..100 {
        // for val in &mut input {
        //     *val = rnd.next_uv_int(2) == 1;
        // }

        for val in 0..10 {
            //print!("Computing");
            // let start = PreciseTime::now();

            for val in &mut input {
                *val = false;
            }
            input[val] = true;

            sp.compute(&input, true);
            //println!(": {:?}", start.to(PreciseTime::now()).num_microseconds().unwrap() as f64 / 1000.0);
            //println!("{:?}", sp.overlaps);
            sp.winner_columns.sort();
            //println!("{:?}", sp.winner_columns);

            let r = classifier.compute(record, val, val as u8, &sp.winner_columns[..], true, true);
            if i == 99 {
                println!("value: {}", val);
                for &(ref step, ref probabilities) in &r {
                    println!("{} {:?}",
                             step,
                             probabilities
                                 .iter()
                                 .enumerate()
                                 .max_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap())
                                 .unwrap());
                }
            }


            record += 1;
        }

    }

}

fn main() {


    let mut images = ImageIter::new("../train-images.idx3-ubyte");
    let mut labels = LabelIter::new("../train-labels.idx1-ubyte");

    let mut test_images = ImageIter::new("../t10k-images.idx3-ubyte");
    let mut test_labels = LabelIter::new("../t10k-labels.idx1-ubyte");


    let mut sp = SpatialPooler::new(vec![28 * 28], vec![40 * 40]);
    sp.potential_radius = 4;
    sp.global_inhibition = true;
    sp.num_active_columns_per_inh_area = 0.3 * sp.num_columns as f64;
    sp.syn_perm_options.active_inc = 0.02; //0.01
    sp.syn_perm_options.inactive_dec = 0.008; //0.008
    sp.syn_perm_options.trim_threshold = 0.005;
    sp.stimulus_threshold = 2.0;
    //sp.syn_perm_options.connected = 0.2;
    //sp.potential_pct = 0.6;
    sp.compability_mode = true;
    sp.init();

    let mut classifier: SDRClassifier<u8> = SDRClassifier::new(vec![0], 0.1, 0.3, sp.num_columns);

    let mut input = vec![false; sp.num_inputs];

    //println!("Training on:{}", images.size);

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

    //println!("Testing on: {}", test_images.size);

    let mut good = 0;
    let mut total = test_images.size;
    for i in 0..test_images.size {

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
