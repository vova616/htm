/// An iterator for `BitVec`.
#[derive(Clone)]
pub struct Iter<'a> {
    bit_vec: &'a BitVec,
    range: std::ops::Range<usize>,
}

impl<'a> Iterator for Iter<'a> {
    type Item = bool;

    #[inline]
    fn next(&mut self) -> Option<bool> {
        // NB: indexing is slow for extern crates when it has to go through &TRUE or &FALSE
        // variables.  get is more direct, and unwrap is fine since we're sure of the range.
        self.range.next().map(|i| self.bit_vec.get(i).unwrap())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.range.size_hint()
    }
}