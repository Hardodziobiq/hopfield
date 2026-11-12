use crate::net::utility::{TestEnd, Utils};

/// LARGE HOPFIELD STRUCT
pub struct LargeHopfield {
    n: usize,
    big_n: usize,
    patterns: Vec<Vec<f32>>,
}

/// LARGE HOPFIELD MAIN FUNCTIONS
impl LargeHopfield {
    pub fn new(patterns: Vec<Vec<f32>>) -> Self {
        let n: usize = Utils::pattern_validate(&patterns);
        let big_n: usize = patterns.len();
        Self { n, big_n, patterns }
    }
    pub fn synchronized_test(&self, probe: &mut Vec<f32>) -> (Vec<Vec<f32>>, TestEnd) {
        self.test_validate(probe);
        self.sync_test(probe)
    }
    pub fn asynchronized_test(&self, probe: &mut Vec<f32>) -> Vec<Vec<f32>> {
        self.test_validate(probe);
        self.async_test(probe)
    }
}

/// LARGE HOPFIELD UTILITY FUNCTIONS
impl LargeHopfield {
    /// VALIDATION
    fn test_validate(&self, probe: &[f32]) {
        if self.n != probe.len() {
            panic!("Given test pattern length doesn't match!");
        }
    }

    fn dot(vec_1: &[f32], vec_2: &[f32], result_vec: &mut [f32], idx: usize) {
        for i in 0..vec_1.len() {
            result_vec[idx] += vec_1[i] * vec_2[i];
        }
    }

    /// SYNC & ASYNC TESTS
    fn sync_test(&self, probe: &mut Vec<f32>) -> (Vec<Vec<f32>>, TestEnd) {
        let mut history: Vec<Vec<f32>> = vec![probe.clone()];
        let mut last_probe: Vec<f32>;
        let mut before_last: Vec<f32> = Vec::new();
        let mut m: Vec<f32> = vec![0.0; self.big_n];
        let mut value: f32;
        loop {
            last_probe = probe.clone();
            m.fill(0.0);
            for (idx, pattern) in self.patterns.iter().enumerate() {
                Self::dot(pattern, probe, &mut m, idx);
            }
            for (idx, cond) in probe.iter_mut().enumerate() {
                value = 0.0;
                for (pattern_idx, pattern) in self.patterns.iter().enumerate() {
                    value += pattern[idx] * m[pattern_idx];
                }
                value -= last_probe[idx] * self.big_n as f32;
                *cond = Utils::sign(value, *cond);
            }
            if *probe == last_probe {
                return (history, TestEnd::Stable);
            }
            history.push(probe.clone());
            if *probe == before_last {
                return (history, TestEnd::Cycle);
            }
            before_last = last_probe;
        }
    }
    fn async_test(&self, probe: &mut Vec<f32>) -> Vec<Vec<f32>> {
        let mut history: Vec<Vec<f32>> = vec![probe.clone()];
        let mut last_probe: Vec<f32>;
        let mut m: Vec<f32> = vec![0.0; self.big_n];
        let mut value: f32;
        loop {
            last_probe = probe.clone();
            m.fill(0.0);
            for (idx, pattern) in self.patterns.iter().enumerate() {
                Self::dot(pattern, probe, &mut m, idx);
            }
            for (idx, cond) in probe.iter_mut().enumerate() {
                value = 0.0;
                for (pattern_idx, pattern) in self.patterns.iter().enumerate() {
                    value += pattern[idx] * m[pattern_idx];
                }
                value -= *cond * self.big_n as f32;
                let old: f32 = *cond;
                let new: f32 = Utils::sign(value, old);
                if new != old {
                    for (pattern_idx, pattern) in self.patterns.iter().enumerate() {
                        m[pattern_idx] += pattern[idx] * (new - old);
                    }
                    *cond = new;
                }
            }
            if *probe == last_probe {
                return history;
            }
            history.push(probe.clone());
        }
    }
}
