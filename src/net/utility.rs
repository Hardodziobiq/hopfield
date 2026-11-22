#![allow(clippy::needless_range_loop)]
use crate::net::hopfield::Hopfield;
use rand::Rng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use std::fs;
use std::path::Path;

/// TEST SETTINGS
pub const SEED: u64 = 42;
pub const OJA_LR: f32 = 0.5;
pub const OJA_EPOCHS: usize = 300;
pub const OJA_TOL: f32 = 1e-6;
pub const NOISE_LEVELS: [f32; 6] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5];
pub const NOISE_REPEATS: usize = 10;
pub const PER_LINE: usize = 13;

/// UTILS FUNCTIONS
pub struct Utils;

/// UTILS MAIN FUNCTIONS
impl Utils {
    pub fn pattern_validate(patterns: &[Vec<f32>]) -> usize {
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
            assert!(
                pattern.iter().all(|&x| x == -1.0 || x == 1.0),
                "Patterns must be bipolar"
            );
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
        Self::read_csv(Path::new(path)).unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn read_csv(path: &Path) -> Result<(Vec<Vec<f32>>, usize, usize), String> {
        let (width, height) = Self::parse_size(path)
            .ok_or_else(|| format!("Expected WIDTHxHEIGHT in {}", path.display()))?;
        let area = width
            .checked_mul(height)
            .filter(|&n| n > 0)
            .ok_or_else(|| format!("Invalid dimensions in {}", path.display()))?;
        let content = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut patterns = Vec::new();
        for (row, line) in content.lines().enumerate() {
            let line: &str = line.trim();
            if line.is_empty() {
                continue;
            }
            let pattern: Vec<f32> = line
                .split(',')
                .map(|value| {
                    value
                        .trim()
                        .parse::<f32>()
                        .ok()
                        .filter(|&x| x == -1.0 || x == 1.0)
                        .ok_or_else(|| format!("{}:{}: expected -1 or 1", path.display(), row + 1))
                })
                .collect::<Result<_, _>>()?;
            if pattern.len() != area {
                return Err(format!(
                    "{}:{}: expected {area} pixels, got {}",
                    path.display(),
                    row + 1,
                    pattern.len()
                ));
            }
            patterns.push(pattern);
        }
        if patterns.is_empty() {
            return Err(format!("{}: empty training set", path.display()));
        }
        Ok((patterns, width, height))
    }
    fn parse_size(path: &Path) -> Option<(usize, usize)> {
        path.file_stem()?
            .to_str()?
            .split(['-', '_', '.'])
            .find_map(|part| {
                let (w, h) = part.split_once('x')?;
                Some((w.parse().ok()?, h.parse().ok()?))
            })
    }

    /// NOISE PATTERNS
    pub fn add_noise_all(
        patterns: &[Vec<f32>],
        fraction: f32,
        rng: &mut impl Rng,
    ) -> Vec<Vec<f32>> {
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
    #[expect(
        clippy::too_many_arguments,
        reason = "Preserve the public rectangle API"
    )]
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
        for y in y0..y0.saturating_add(h).min(height) {
            for x in x0..x0.saturating_add(w).min(width) {
                occluded[y * width + x] = value;
            }
        }
        occluded
    }

    /// DATASETS
    pub fn load_datasets(dir: &str) -> Vec<Dataset> {
        let mut paths: Vec<String> = Vec::new();
        for entry in fs::read_dir(dir).expect("Cannot read data dir!") {
            let path: String = entry
                .expect("Bad dir entry!")
                .path()
                .to_string_lossy()
                .to_string();
            if path.ends_with(".csv") {
                paths.push(path);
            }
        }
        paths.sort();

        let mut datasets: Vec<Dataset> = Vec::new();
        for path in paths {
            let (patterns, width, height) = Self::load_patterns(&path);
            let name: String = Path::new(&path)
                .file_name()
                .expect("Bad file name!")
                .to_string_lossy()
                .to_string();
            datasets.push(Dataset {
                name,
                patterns,
                width,
                height,
            });
        }
        datasets
    }
    pub fn find_dataset<'a>(datasets: &'a [Dataset], name: &str) -> &'a Dataset {
        for ds in datasets {
            if ds.name == name {
                return ds;
            }
        }
        panic!("Dataset {} not found!", name);
    }

    /// CONSOLE OUTPUT
    pub fn print_header(title: &str, description: &str) {
        println!("\n{}", "=".repeat(71));
        println!("{}", title);
        println!("{}", "=".repeat(71));
        println!("{}\n", description);
    }
    pub fn print_cells(label: &str, cells: &[String]) {
        let indent: String = " ".repeat(label.chars().count());
        for (chunk_idx, chunk) in cells.chunks(PER_LINE).enumerate() {
            if chunk_idx == 0 {
                print!("{}", label);
            } else {
                print!("{}", indent);
            }
            for cell in chunk {
                print!("{:>5}", cell);
            }
            println!();
        }
    }
    pub fn print_levels_header() {
        print!("    {:<12}", "");
        for level in NOISE_LEVELS {
            print!("{:>7}", format!("{:.0}%", level * 100.0));
        }
        println!();
    }
    pub fn print_rate_row(label: &str, values: &[f32]) {
        print!("    {:<12}", label);
        for value in values {
            print!("{:>7.2}", value);
        }
        println!();
    }

    /// PATTERN SIMILARITY (|x_p · x_q| / n: 0 = different, 1 = identical or negative)
    pub fn similarity(a: &[f32], b: &[f32]) -> f32 {
        let mut dot: f32 = 0.0;
        for i in 0..a.len() {
            dot += a[i] * b[i];
        }
        (dot / a.len() as f32).abs()
    }
    pub fn mean_similarity(patterns: &[Vec<f32>]) -> f32 {
        let mut sum: f32 = 0.0;
        let mut pairs: usize = 0;
        for p in 0..patterns.len() {
            for q in (p + 1)..patterns.len() {
                sum += Self::similarity(&patterns[p], &patterns[q]);
                pairs += 1;
            }
        }
        if pairs == 0 { 0.0 } else { sum / pairs as f32 }
    }
    /// (p, q, similarity) of the most similar pair
    pub fn most_similar_pair(patterns: &[Vec<f32>]) -> (usize, usize, f32) {
        let mut best: (usize, usize, f32) = (0, 0, 0.0);
        for p in 0..patterns.len() {
            for q in (p + 1)..patterns.len() {
                let sim: f32 = Self::similarity(&patterns[p], &patterns[q]);
                if sim > best.2 {
                    best = (p, q, sim);
                }
            }
        }
        best
    }

    /// STABILITY (pixels changed in the first sync step, 0 = stable pattern)
    pub fn unstable_pixels(net: &Hopfield, pattern: &[f32]) -> usize {
        let mut probe: Vec<f32> = pattern.to_vec();
        net.step(&mut probe, Mode::Sync);
        probe.iter().zip(pattern).filter(|(a, b)| a != b).count()
    }
    pub fn count_stable(net: &Hopfield, patterns: &[Vec<f32>]) -> usize {
        let mut count: usize = 0;
        for pattern in patterns {
            if Self::unstable_pixels(net, pattern) == 0 {
                count += 1;
            }
        }
        count
    }

    /// RECALL (recovery from damaged input)
    /// Async always ends in a stable state.
    pub fn recall(net: &Hopfield, probe: &mut [f32], mode: Mode) -> TestEnd {
        net.run(probe, mode).1
    }
    /// Damages every pattern with `damage` NOISE_REPEATS times and recalls it.
    /// Returns (fraction of exact recoveries, mean fraction of correct pixels).
    pub fn recovery_rate(
        net: &Hopfield,
        patterns: &[Vec<f32>],
        mode: Mode,
        rng: &mut StdRng,
        damage: &dyn Fn(&[f32], &mut StdRng) -> Vec<f32>,
    ) -> (f32, f32) {
        let mut ok: usize = 0;
        let mut total: usize = 0;
        let mut pixel_sum: f32 = 0.0;
        for _ in 0..NOISE_REPEATS {
            for original in patterns {
                let mut probe: Vec<f32> = damage(original, rng);
                let end: TestEnd = Self::recall(net, &mut probe, mode);
                if end == TestEnd::Stable && probe == *original {
                    ok += 1;
                }
                let mut same: usize = 0;
                for i in 0..original.len() {
                    if probe[i] == original[i] {
                        same += 1;
                    }
                }
                pixel_sum += same as f32 / original.len() as f32;
                total += 1;
            }
        }
        (ok as f32 / total as f32, pixel_sum / total as f32)
    }
    /// recovery_rate for every level in NOISE_LEVELS: (exact, pixels)
    pub fn noise_test_stats(
        net: &Hopfield,
        patterns: &[Vec<f32>],
        mode: Mode,
        rng: &mut StdRng,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut exact: Vec<f32> = Vec::new();
        let mut pixels: Vec<f32> = Vec::new();
        for level in NOISE_LEVELS {
            let damage = |p: &[f32], r: &mut StdRng| Self::add_noise(p, level, r);
            let (e, px) = Self::recovery_rate(net, patterns, mode, rng, &damage);
            exact.push(e);
            pixels.push(px);
        }
        (exact, pixels)
    }
}

/// TEST END ENUM
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TestEnd {
    Stable,
    Cycle,
}

/// RECALL MODE
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Mode {
    Sync,
    Async,
}

/// ONE LOADED SET OF PATTERNS
pub struct Dataset {
    pub name: String,
    pub patterns: Vec<Vec<f32>>,
    pub width: usize,
    pub height: usize,
}
