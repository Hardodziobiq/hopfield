use std::fs;
use std::path::Path;
use rand::seq::SliceRandom;
use rand::Rng;

/// UTILS FUNCTIONS
pub struct Utils;

/// UTILS MAIN FUNCTIONS
impl Utils {
    pub fn pattern_validate(patterns: &Vec<Vec<f32>>) -> usize {
        if patterns.is_empty() {
            panic!("Empty patterns!");
        }
        if patterns[0].is_empty() {
            panic!("First pattern empty!");
        }
        let pattern_len: usize = patterns[0].len();
        for pattern in patterns {
            if pattern.len() != pattern_len {
                panic!("Different patterns length!");
            }
        }
        pattern_len
    }
    pub fn sign(value: f32, old: f32) -> f32 {
        if 0.0 < value {
            1.0
        } else if value < 0.0 {
            -1.0
        } else {
            old
        }
    }

    /// PARSING FILE (PATTERN, WIDTH, HEIGHT)
    pub fn load_patterns(path: &str) -> (Vec<Vec<f32>>, usize, usize) {
        let (width, height) = Self::parse_size(path);
        let content: String = fs::read_to_string(path).expect("Cannot read file!");
        let mut patterns: Vec<Vec<f32>> = Vec::new();
        for line in content.lines() {
            let line: &str = line.trim();
            if line.is_empty() {
                continue;
            }
            let mut pattern: Vec<f32> = Vec::new();
            for value in line.split(',') {
                pattern.push(value.trim().parse::<f32>().expect("Not a number!"));
            }
            if pattern.len() != width * height {
                panic!("Pattern length {} doesn't match {}x{}!", pattern.len(), width, height);
            }
            patterns.push(pattern);
        }
        (patterns, width, height)
    }
    fn parse_size(path: &str) -> (usize, usize) {
        let stem: &str = Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .expect("Bad file name!");
        for part in stem.split(|c: char| c == '-' || c == '_' || c == '.') {
            if let Some((w, h)) = part.split_once('x') {
                if let (Ok(width), Ok(height)) = (w.parse::<usize>(), h.parse::<usize>()) {
                    return (width, height);
                }
            }
        }
        panic!("Cannot find WIDTHxHEIGHT in file name: {}", path);
    }

    /// NOISE PATTERNS
    pub fn add_noise_all(patterns: &[Vec<f32>], fraction: f32, rng: &mut impl Rng) -> Vec<Vec<f32>> {
        let mut noisy: Vec<Vec<f32>> = Vec::new();
        for pattern in patterns {
            noisy.push(Self::add_noise(pattern, fraction, rng));
        }
        noisy
    }
    /// Random pixels change values
    /// fraction 0.2 => 20% pixels change values
    pub fn add_noise(pattern: &[f32], fraction: f32, rng: &mut impl Rng) -> Vec<f32> {
        let mut noisy: Vec<f32> = pattern.to_vec();
        let flips: usize = (fraction * pattern.len() as f32).round() as usize;
        let mut indices: Vec<usize> = (0..pattern.len()).collect();
        indices.shuffle(rng);
        for &i in indices.iter().take(flips) {
            noisy[i] = -noisy[i];
        }
        noisy
    }
    /// Creates rectangle on pattern with given value
    /// (x0, y0) left upper corner
    /// w - width, h - height
    pub fn occlude(
        pattern: &[f32],
        width: usize,
        height: usize,
        x0: usize,
        y0: usize,
        w: usize,
        h: usize,
        value: f32,
    ) -> Vec<f32> {
        let mut occluded: Vec<f32> = pattern.to_vec();
        for y in y0..(y0 + h).min(height) {
            for x in x0..(x0 + w).min(width) {
                occluded[y * width + x] = value;
            }
        }
        occluded
    }
}

/// TEST END ENUM
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TestEnd {
    Stable,
    Cycle,
}
