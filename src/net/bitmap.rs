use std::fs;
use std::path::Path;

use crate::net::utility::Utils;

pub struct Bitmaps {
    pub width: usize,
    pub height: usize,
    pub patterns: Vec<Vec<f32>>,
}

impl Bitmaps {
    /// Accept a CSV, a plain PBM, or a directory of plain PBMs.
    pub fn load(path: &Path) -> Result<Self, String> {
        if path.is_dir() {
            let entries = fs::read_dir(path).map_err(|e| format!("{}: {e}", path.display()))?;
            let mut files = Vec::new();
            for entry in entries {
                let file = entry.map_err(|e| e.to_string())?.path();
                if file
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("pbm"))
                {
                    files.push(file);
                }
            }
            files.sort();
            let mut files = files.iter();
            let first = files
                .next()
                .ok_or_else(|| format!("{}: no PBM files", path.display()))?;
            let mut bitmaps = Self::read_pbm(first)?;
            for file in files {
                let mut bitmap = Self::read_pbm(file)?;
                if (bitmap.width, bitmap.height) != (bitmaps.width, bitmaps.height) {
                    return Err(format!(
                        "{}: dimensions differ from {}",
                        file.display(),
                        first.display()
                    ));
                }
                bitmaps.patterns.append(&mut bitmap.patterns);
            }
            return Ok(bitmaps);
        }
        if path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("pbm"))
        {
            Self::read_pbm(path)
        } else {
            let (patterns, width, height) = Utils::read_csv(path)?;
            Ok(Self {
                width,
                height,
                patterns,
            })
        }
    }

    fn read_pbm(path: &Path) -> Result<Self, String> {
        let fail = |message: &str| format!("{}: {message}", path.display());
        let data = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut tokens = data.lines().flat_map(|line| {
            line.split('#')
                .next()
                .unwrap_or_default()
                .split_ascii_whitespace()
        });
        if tokens.next() != Some("P1") {
            return Err(fail("expected a plain PBM (P1)"));
        }
        let mut dimension = || {
            tokens
                .next()
                .and_then(|s| s.parse::<usize>().ok())
                .filter(|&n| n > 0)
        };
        let width = dimension().ok_or_else(|| fail("invalid width"))?;
        let height = dimension().ok_or_else(|| fail("invalid height"))?;
        let area = width
            .checked_mul(height)
            .ok_or_else(|| fail("invalid dimensions"))?;

        let pattern = tokens
            .flat_map(str::bytes)
            .map(|pixel| match pixel {
                b'1' => Ok(1.0),
                b'0' => Ok(-1.0),
                _ => Err(fail("expected a 0 or 1 pixel")),
            })
            .collect::<Result<Vec<_>, _>>()?;
        if pattern.len() != area {
            return Err(fail("pixel count does not match the dimensions"));
        }
        Ok(Self {
            width,
            height,
            patterns: vec![pattern],
        })
    }
}
