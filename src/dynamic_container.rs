
use std::default::Default;
use std::clone::Clone;
use std;
use numext::ClipExt;

#[derive(Debug)]
pub struct DynamicContainer<T> {
    container: Vec<T>,
    count_children: Vec<usize>,
    max_children: usize,
}

impl<T> DynamicContainer<T> where T : Default + Clone  {
    pub fn new(parents: usize, max_children: usize) -> DynamicContainer<T> {
        DynamicContainer {
            container: vec![Default::default(); parents * max_children],
            count_children: vec![0; parents],
            max_children: max_children,
        }
    }

    pub fn insert(&mut self, parent: usize, child: T ) -> usize {
        let mut count = self.count_children[parent];
        if count < self.max_children {
            self.container[parent * self.max_children + count] = child;
            count += 1;
            self.count_children[parent] = count;
            count
        } else {
            panic!("tryring to insert more children than max_children");
        }
    }

    pub fn sort_pivot_children<F>(&mut self, parent: usize, compare: F) -> usize where F: Fn(&T) -> bool {
        let range = self.children_range(parent);
        let arr = &mut self.container;

        let mut pivot = range.start;
        for i in range.clone() {
            if compare(&arr[i]) {
                if (pivot != i) {
                    arr.swap(i, pivot);
                }
                pivot += 1;
            }
        }
        pivot - range.start
    }

    pub fn children_range(&self, parent: usize) -> std::ops::Range<usize> {
        let start_index = parent * self.max_children;
        let end_index = start_index + self.count_children[parent];
        start_index..end_index
    }

    pub fn children_range_sized(&self, parent: usize, size: usize) -> std::ops::Range<usize> {
        if size > self.count_children[parent] {
            panic!("taking size bigger than contained")
        }
        let start_index = parent * self.max_children;
        let end_index = start_index + size;
        start_index..end_index
    }

    pub fn children(&self, parent: usize) -> &[T] {
        let range = self.children_range(parent);
        &self.container[range]
    }

    pub fn children_sized(&self, parent: usize, size: usize) -> &[T] {
        let range = self.children_range_sized(parent, size);
        &self.container[range]
    }

    pub fn children_mut(&mut self, parent: usize) -> &mut [T] {
        let range = self.children_range(parent);
        &mut self.container[range]
    }

    pub fn children_sized_mut(&mut self, parent: usize, size: usize) -> &mut [T] {
        let range = self.children_range_sized(parent, size);
        &mut self.container[range]
    }
   
}
