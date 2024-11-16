use core::slice;
use std::{iter::FusedIterator, mem::MaybeUninit};

#[derive(Debug)]
pub struct RingBuffer<T, const N: usize> {
    data: [MaybeUninit<T>; N],
    write_loc: usize,
    read_loc: usize,
    capacity: usize, // TODO: maybe can read this data compile time?
}

// impl<T, const N: usize, Idx> Index<Idx> for RingBuffer<T, N>
// where
//     Idx: SliceIndex<[T], Output = T> + Into<usize>,
// {
//     type Output = T;

//     #[inline(always)]
//     fn index(&self, index: Idx) -> &Self::Output {
//         let i = index.into() % self.max_entries;
//         assert!(i >= self.read_loc);
//         assert!(self.write_loc > i);
//         let item = &unsafe { self.data.assume_init_ref() }[i];
//         item
//     }
// }

// impl<T, const N: usize> Iterator for RingBuffer<T, N> {
//     type Item = T;

//     fn next(&mut self) -> Option<Self::Item> {
//         self.pop()
//     }
// }

impl<T, const N: usize> RingBuffer<T, N> {
    pub fn new() -> Self {
        let maybe_uninit_data: [MaybeUninit<T>; N] = [const { MaybeUninit::<T>::uninit() }; N];
        let write_loc = 0;
        let read_loc = 0;
        let max_entries = N;

        Self {
            data: maybe_uninit_data,
            write_loc,
            read_loc,
            capacity: max_entries,
        }
    }

    pub fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut buffer = Self::new();
        for (i, item) in iter.into_iter().enumerate() {
            if i < N {
                buffer.push(item);
            } else {
                break; // Stop if we exceed the buffer capacity
            }
        }
        buffer
    }

    pub fn len(&self) -> usize {
        self.write_loc - self.read_loc
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn is_empty(&self) -> bool {
        self.len() <= 0
    }

    pub fn space_left(&self) -> usize {
        self.capacity - self.len()
    }

    pub fn clear(&mut self) {
        while !self.is_empty() {
            self.pop_front();
        }
    }

    pub fn push(&mut self, entry: T) {
        assert_eq!(self.len() < self.capacity(), true, "RingBuffer Overflow");

        self.data[self.write_loc % self.capacity].write(entry);
        self.write_loc += 1;
    }

    pub fn pop_front(&mut self) -> Option<T> {
        if self.len() <= 0 {
            return None;
        }

        let read_index = self.read_loc % self.capacity;

        let read_value = unsafe { self.data[read_index].assume_init_read() };
        self.read_loc += 1;

        Some(read_value)
    }

    pub fn peek_front(&self) -> Option<&T> {
        if self.len() <= 0 {
            return None;
        }

        let data = unsafe { self.data[self.read_loc % self.capacity].assume_init_ref() };

        Some(data)
    }

    #[inline]
    #[allow(dead_code)]
    pub fn get_ptr(&self) -> *const T {
        self.data.as_ptr().cast()
    }

    #[inline]
    #[allow(dead_code)]
    pub fn get_mut_ptr(&mut self) -> *const T {
        self.data.as_mut_ptr().cast()
    }

    #[inline]
    #[allow(dead_code)]
    pub fn get_relative(&self, index: usize) -> Option<&T> {
        if index < self.len() {
            let read_index = (self.read_loc + index) % self.capacity;
            let val_ref = unsafe { self.data[read_index].assume_init_ref() };
            Some(val_ref)
        } else {
            None
        }
    }

    #[inline]
    #[allow(dead_code)]
    pub fn get_relative_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < self.len() {
            let read_index = (self.read_loc + index) % self.capacity;
            let val_ref = unsafe { self.data[read_index].assume_init_mut() };
            Some(val_ref)
        } else {
            None
        }
    }

    #[inline]
    pub fn iter(&self) -> Iter<T> {
        Iter {
            buffer: unsafe { slice::from_raw_parts(self.data.as_ptr().cast::<T>(), self.capacity) },
            buffer_capacity: self.capacity,
            read_loc: self.read_loc,
            write_loc: self.write_loc,
        }
    }

    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<T> {
        IterMut {
            buffer: unsafe {
                slice::from_raw_parts_mut(self.data.as_mut_ptr().cast::<T>(), self.capacity)
            },
            buffer_capacity: self.capacity,
            read_loc: self.read_loc,
            write_loc: self.write_loc,
        }
    }
}

impl<T, const N: usize> Drop for RingBuffer<T, N> {
    fn drop(&mut self) {
        while !self.is_empty() {
            drop(self.pop_front());
        }
    }
}

pub struct IntoIter<T, const N: usize> {
    inner: RingBuffer<T, N>,
}

impl<T, const N: usize> Iterator for IntoIter<T, N> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.pop_front()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.inner.len(), None)
    }
}

pub struct Iter<'a, T: 'a> {
    buffer: &'a [T],
    buffer_capacity: usize,
    read_loc: usize,
    write_loc: usize,
}

impl<T> Iter<'_, T> {
    fn len(&self) -> usize {
        self.write_loc - self.read_loc
    }
}

impl<'a, T> Iterator for Iter<'a, T>
where
    T: 'a,
{
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.len() > 0 {
            let current_index = self.read_loc;
            self.read_loc = (self.read_loc + 1) % self.buffer_capacity;

            unsafe {
                let elem = self.buffer.get_unchecked(current_index);
                Some(elem)
            }
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), None)
    }
}

// impl<T> ExactSizeIterator for Iter<'_, T> {}
// impl<T> FusedIterator for Iter<'_, T> {}

pub struct IterMut<'a, T: 'a> {
    buffer: &'a mut [T],
    buffer_capacity: usize,
    read_loc: usize,
    write_loc: usize,
}

impl<T> IterMut<'_, T> {
    fn len(&self) -> usize {
        self.write_loc - self.read_loc
    }
}

impl<'a, T> Iterator for IterMut<'a, T>
where
    T: 'a,
{
    type Item = &'a mut T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.len() > 0 {
            let current_index = self.read_loc;
            self.read_loc = (self.read_loc + 1) % self.buffer_capacity;

            unsafe {
                let elem = self.buffer.get_unchecked_mut(current_index);
                // and now for some black magic
                // the std stuff does this too, but afaik this breaks the borrow checker
                Some(&mut *(elem as *mut T))
            }
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), None)
    }
}

// impl<T> ExactSizeIterator for IterMut<'_, T> {}
// impl<T> FusedIterator for IterMut<'_, T> {}

impl<T, const N: usize> IntoIterator for RingBuffer<T, N> {
    type Item = T;
    type IntoIter = IntoIter<T, N>;

    /// Consumes the `RingBuffer` into a front-to-back iterator yielding elements by
    /// value.
    fn into_iter(self) -> Self::IntoIter {
        IntoIter { inner: self }
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a RingBuffer<T, N> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a mut RingBuffer<T, N> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rand::RngCore;

    use super::*;

    #[test]
    fn test_ringbuffer_unused() {
        println!("usize BEFORE");
        let mut _ringbuffer: RingBuffer<usize, 160> = RingBuffer::new();
        println!("usize AFTER");
    }

    #[test]
    fn test_ringbuffer_usize() {
        const RINGBUFFER_SIZE: usize = 10;
        println!("Buffersize: {RINGBUFFER_SIZE}, 0 spot left");
        let mut buffer: RingBuffer<usize, RINGBUFFER_SIZE> = RingBuffer::new();

        assert_eq!(buffer.space_left(), RINGBUFFER_SIZE);

        const PUSH_NUM: usize = 10;
        let mut push_counter: usize = 0;
        while push_counter < PUSH_NUM {
            let to_push = push_counter;
            println!("Push: {}", to_push);
            buffer.push(to_push);
            push_counter += 1;
            assert_eq!(buffer.len(), push_counter);
        }

        let mut pop_counter = push_counter;
        while buffer.len() >= 1 {
            let popped = buffer.pop_front().unwrap();
            println!("Pop: {}", popped);
            pop_counter -= 1;
            assert_eq!(buffer.len(), pop_counter);
        }
        assert_eq!(buffer.len(), 0);

        const RINGBUFFER_SIZE_2: usize = 10;
        println!("Buffersize: {RINGBUFFER_SIZE_2}, 1 spot left");
        let mut buffer_2: RingBuffer<usize, RINGBUFFER_SIZE_2> = RingBuffer::new();

        const PUSH_NUM_2: usize = 9;
        let mut push_counter = 0;
        while push_counter < PUSH_NUM_2 {
            let to_push = push_counter;
            println!("Push: {}", to_push);
            buffer_2.push(to_push);
            push_counter += 1;
            assert_eq!(buffer_2.len(), push_counter);
        }

        assert_eq!(buffer_2.space_left(), 1);

        let mut pop_counter = push_counter;
        while buffer_2.len() >= 1 {
            let popped = buffer_2.pop_front().unwrap();
            println!("Pop: {}", popped);
            pop_counter -= 1;
            assert_eq!(buffer_2.len(), pop_counter);
        }

        assert_eq!(buffer_2.len(), 0);

        const RINGBUFFER_SIZE_3: usize = 1572;
        println!("Buffersize: {RINGBUFFER_SIZE_3}");
        let mut buffer_3: RingBuffer<usize, RINGBUFFER_SIZE_3> = RingBuffer::new();

        const NUM_RANDOM_PUSH_POPS: usize = 898;
        let mut random_bits = [0u8; NUM_RANDOM_PUSH_POPS];
        rand::thread_rng().fill_bytes(&mut random_bits);
        let random_elements: [usize; NUM_RANDOM_PUSH_POPS] = [0; NUM_RANDOM_PUSH_POPS];
        rand::thread_rng().fill_bytes(&mut random_bits);

        let mut counter = 0;
        for ele in random_elements {
            let random_bit: bool = random_bits[counter] != 0;
            if random_bit && buffer_3.len() >= 1 {
                buffer_3.pop_front();
            } else {
                buffer_3.push(ele);
            }

            counter += 1;
        }

        buffer_3.clear();
        assert_eq!(buffer_3.len(), 0);
        const PUSH_NUM_3: usize = 5;
        push_counter = 0;
        while push_counter < PUSH_NUM_3 {
            let to_push = push_counter;
            println!("Push: {}", to_push);
            buffer_3.push(to_push);
            push_counter += 1;
        }

        pop_counter = push_counter;
        while pop_counter >= 1 {
            let popped = buffer_3.pop_front().unwrap();
            println!("Pop: {}", popped);
            pop_counter -= 1;
        }
        assert_eq!(buffer_3.len(), 0);
    }

    #[test]
    #[should_panic]
    fn test_ringbuffer_usize_overflow() {
        const RINGBUFFER_SIZE: usize = 10;
        let mut buffer: RingBuffer<usize, RINGBUFFER_SIZE> = RingBuffer::new();

        const PUSH_NUM: usize = 11;
        let mut push_counter: usize = 0;
        while push_counter < PUSH_NUM {
            let to_push = push_counter;
            println!("Push: {}", to_push);
            buffer.push(to_push);
            push_counter += 1;
            assert_eq!(buffer.len(), push_counter);
        }

        let mut pop_counter = push_counter;
        while buffer.len() >= 1 {
            let popped = buffer.pop_front().unwrap();
            println!("Pop: {}", popped);
            pop_counter -= 1;
            assert_eq!(buffer.len(), pop_counter);
        }

        assert_eq!(buffer.len(), 0);
    }

    #[test]
    #[should_panic]
    fn test_ringbuffer_usize_pop_before_push() {
        const RINGBUFFER_SIZE: usize = 10;
        let mut buffer: RingBuffer<usize, RINGBUFFER_SIZE> = RingBuffer::new();
        buffer.pop_front().unwrap();
    }

    #[test]
    fn test_ringbuffer_f32_iterator() {
        const RINGBUFFER_SIZE: usize = 10;
        println!("Buffer size {RINGBUFFER_SIZE}: loop enumerate");
        let mut buffer: RingBuffer<Arc<f32>, RINGBUFFER_SIZE> = RingBuffer::new();

        buffer.push(Arc::new(32.0));
        buffer.push(Arc::new(1100.0));
        buffer.push(Arc::new(13320.0));
        for (i, f) in buffer.iter().enumerate() {
            println!("[{i}]: {f}");
        }

        const RINGBUFFER_SIZE_2: usize = 10;
        println!("Buffer size {RINGBUFFER_SIZE_2}: into_iter()");
        let mut buffer: RingBuffer<Arc<f32>, RINGBUFFER_SIZE_2> = RingBuffer::new();

        buffer.push(Arc::new(32.0));
        buffer.push(Arc::new(1100.0));
        buffer.push(Arc::new(13320.0));
        buffer.push(Arc::new(0.0));

        {
            let mut iter = buffer.iter();
            let _a = iter.next().unwrap();
            let _b = iter.next().unwrap();
            let _b = iter.next().unwrap();
            let _b = iter.next().unwrap();
        }

        buffer.push(Arc::new(32.0));
        buffer.push(Arc::new(1100.0));
        buffer.push(Arc::new(13320.0));
        buffer.push(Arc::new(0.0));

        buffer.get_relative(buffer.len() - 1).unwrap();
    }

    #[test]
    fn test_ringbuffer_f32_underflow() {
        const RINGBUFFER_SIZE: usize = 10;
        let mut buffer: RingBuffer<f32, RINGBUFFER_SIZE> = RingBuffer::new();

        buffer.push(32.0);
        buffer.push(1100.0);
        buffer.push(13320.0);
        buffer.push(0.0);

        for a in buffer.iter() {
            println!("{}", a.atan());
        }
    }

    #[test]
    fn test_ringbuffer_f32_arc() {
        const RINGBUFFER_SIZE: usize = 10;
        let mut buffer: RingBuffer<Arc<f32>, RINGBUFFER_SIZE> = RingBuffer::new();

        buffer.push(Arc::new(32.0));
        buffer.push(Arc::new(1100.0));
        buffer.push(Arc::new(13320.0));
        buffer.push(Arc::new(0.0));

        for a in buffer.iter() {
            println!("{}", a);
        }
    }

    #[test]
    fn test_ringbuffer_relative() {
        const RINGBUFFER_SIZE: usize = 10;
        let mut buffer: RingBuffer<usize, RINGBUFFER_SIZE> = RingBuffer::new();
        for i in 0..120 {
            if buffer.space_left() <= 0 {
                buffer.clear();
            }
            buffer.push(i);
        }

        let _a = buffer.get_relative(9).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_ringbuffer_relative_should_panic() {
        const RINGBUFFER_SIZE: usize = 10;
        let mut buffer: RingBuffer<usize, RINGBUFFER_SIZE> = RingBuffer::new();
        for i in 0..123 {
            if buffer.space_left() <= 0 {
                buffer.clear();
            }
            buffer.push(i);
        }

        buffer.get_relative(3).unwrap();
    }

    #[test]
    fn test_ringbuffer_from_iter() {
        let initial_elements = vec![1, 2, 3, 4, 5];
        let mut buffer: RingBuffer<_, 10> = RingBuffer::from_iter(initial_elements);

        assert_eq!(buffer.len(), 5);
        assert_eq!(buffer.pop_front(), Some(1));
        assert_eq!(buffer.pop_front(), Some(2));
        assert_eq!(buffer.pop_front(), Some(3));
        assert_eq!(buffer.pop_front(), Some(4));
        assert_eq!(buffer.pop_front(), Some(5));
        assert_eq!(buffer.pop_front(), None);
    }
}
