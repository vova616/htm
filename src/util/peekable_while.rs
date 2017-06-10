use std::iter;

pub struct PeekableWhile<'a, I : Iterator + 'a, P: FnMut(&I::Item) -> bool> {
    iter: &'a mut iter::Peekable<I>,
    predicate: P,
}

pub trait PeekableWhileTrait<'a, I : Iterator,P: FnMut(&I::Item) -> bool> {
    fn take_while_peek(&'a mut self, predicate:P) -> PeekableWhile<'a, I,P>;
} 

impl<'a, I: Iterator, P: FnMut(&I::Item) -> bool> PeekableWhileTrait<'a, I,P> for iter::Peekable<I> {
    fn take_while_peek(&'a mut self, predicate:P) -> PeekableWhile<'a, I,P> {
        PeekableWhile{ iter: self, predicate: predicate}
    }
}

impl<'a, I : Iterator + 'a, P: FnMut(&I::Item) -> bool> PeekableWhile<'a, I, P>  {
    pub fn new(peekable: &mut iter::Peekable<I>, predicate: P ) -> PeekableWhile<I,P> {
        PeekableWhile{ iter: peekable, predicate: predicate}
    }

    #[inline]
    pub fn peek(&mut self) -> Option<&I::Item> {
        self.iter.peek()
    }

    #[inline]
    pub fn drain(&mut self)  {
       loop {
           match self.next() {
               None => break,
               _ => {},
           }
       }
    }
}

impl<'a, I: Iterator, P> Iterator for PeekableWhile<'a, I, P>
    where P: FnMut(&I::Item) -> bool
{
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<I::Item> {
        let r = match self.iter.peek() {
            Some(peek) => if (self.predicate)(&peek) { true } else { false },
            None => false,
        };
        if r {
            self.iter.next()
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, upper) = self.iter.size_hint();
        (0, upper) // can't know a lower bound, due to the predicate
    }
}


