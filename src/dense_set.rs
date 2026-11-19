//! Fixed-width sets of dense ids, one bit per id.

#[derive(Clone, PartialEq, Eq)]
pub(crate) struct DenseSet {
    words: Vec<u64>,
}

impl DenseSet {
    pub(crate) fn new(width: usize) -> Self {
        Self {
            words: vec![0; width.div_ceil(64)],
        }
    }

    pub(crate) fn insert(&mut self, id: usize) {
        self.words[id / 64] |= 1 << (id % 64);
    }

    pub(crate) fn remove(&mut self, id: usize) {
        self.words[id / 64] &= !(1 << (id % 64));
    }

    pub(crate) fn contains(&self, id: usize) -> bool {
        self.words
            .get(id / 64)
            .is_some_and(|word| word & (1 << (id % 64)) != 0)
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.words
            .iter()
            .enumerate()
            .flat_map(|(word_index, &word)| {
                let mut bits = word;
                std::iter::from_fn(move || {
                    if bits == 0 {
                        return None;
                    }
                    let bit = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    Some(word_index * 64 + bit)
                })
            })
    }

    pub(crate) fn union_with(&mut self, other: &Self) {
        for (word, other) in self.words.iter_mut().zip(&other.words) {
            *word |= other;
        }
    }

    pub(crate) fn intersection_count(&self, other: &Self) -> usize {
        self.words
            .iter()
            .zip(&other.words)
            .map(|(a, b)| (a & b).count_ones() as usize)
            .sum()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::DenseSet;
    use std::collections::HashSet;

    fn step(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *state >> 33
    }

    #[test]
    fn dense_set_agrees_with_a_hash_set_model() {
        let width = 200;
        let mut dense = DenseSet::new(width);
        let mut model = HashSet::new();
        let mut state = 7;
        for _ in 0..5000 {
            let id = step(&mut state) as usize % width;
            if step(&mut state) % 2 == 0 {
                dense.insert(id);
                model.insert(id);
            } else {
                dense.remove(id);
                model.remove(&id);
            }
            assert_eq!(dense.contains(id), model.contains(&id));
        }
        let mut listed: Vec<usize> = dense.iter().collect();
        listed.sort_unstable();
        let mut expected: Vec<usize> = model.iter().copied().collect();
        expected.sort_unstable();
        assert_eq!(listed, expected);
    }

    #[test]
    fn union_and_intersection_agree_with_the_model() {
        let width = 130;
        let mut state = 11;
        for _ in 0..200 {
            let mut a = DenseSet::new(width);
            let mut b = DenseSet::new(width);
            let mut model_a = HashSet::new();
            let mut model_b = HashSet::new();
            for _ in 0..40 {
                let (x, y) = (
                    step(&mut state) as usize % width,
                    step(&mut state) as usize % width,
                );
                a.insert(x);
                model_a.insert(x);
                b.insert(y);
                model_b.insert(y);
            }
            assert_eq!(
                a.intersection_count(&b),
                model_a.intersection(&model_b).count()
            );
            let mut union = a.clone();
            union.union_with(&b);
            let mut listed: Vec<usize> = union.iter().collect();
            listed.sort_unstable();
            let mut expected: Vec<usize> = model_a.union(&model_b).copied().collect();
            expected.sort_unstable();
            assert_eq!(listed, expected);
        }
    }
}
