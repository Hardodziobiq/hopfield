use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use hopfield::net::bitmap::Bitmaps;
use hopfield::net::bitmap_network::train_bitmap;
use hopfield::net::utility::{Mode, SEED, TestEnd, Utils};
use rand::SeedableRng;
use rand::rngs::StdRng;

pub const USAGE: &str =
    "viz2 DATASET INDEX hebb|oja sync|async NOISE_PERCENT OUTPUT.html [INPUT]";

/// About this many frames are generated for one asynchronous pass.
const FRAMES_PER_PASS: usize = 25;

/// One frame of the animation: network state and its description.
struct Frame {
    state: Vec<f32>,
    label: String,
}

/// Step-by-step visualization. In async mode every pass is split into
/// frames of a few rows, so the repair of the image is visible inside a pass.
/// Next to the network state the page shows the clean pattern.
pub fn run(args: &[String]) -> Result<(), String> {
    let [dataset, index, rule, update_mode, noise, output, rest @ ..] = args else {
        return Err(USAGE.to_owned());
    };
    if rest.len() > 1 {
        return Err(USAGE.to_owned());
    }
    let index: usize = index
        .parse()
        .map_err(|_| "Index must be a nonnegative integer")?;
    let noise: f32 = noise
        .parse()
        .map_err(|_| "Noise must be a number between 0 and 100")?;
    if !(0.0..=100.0).contains(&noise) {
        return Err("Noise must be a number between 0 and 100".into());
    }
    let mode = match update_mode.as_str() {
        "sync" => Mode::Sync,
        "async" => Mode::Async,
        _ => return Err("Update mode must be sync or async".into()),
    };
    let (oja, label) = match rule.as_str() {
        "hebb" => (false, "Hebb"),
        "oja" => (true, "Oja"),
        _ => return Err("Learning rule must be hebb or oja".into()),
    };
    let output = Path::new(output);
    if !output
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("html"))
    {
        return Err("Output filename must end in .html".into());
    }

    let bitmaps = Bitmaps::load(Path::new(dataset))?;
    let (width, height) = (bitmaps.width, bitmaps.height);
    let input_bitmaps = rest
        .first()
        .map(|path| Bitmaps::load(Path::new(path)))
        .transpose()?;
    let input = match input_bitmaps {
        Some(ref input) if (input.width, input.height) != (width, height) => {
            return Err("Input and training images must have the same dimensions".into());
        }
        Some(ref input) => &input.patterns,
        None => &bitmaps.patterns,
    };
    let pattern = input
        .get(index)
        .ok_or_else(|| format!("Available indices: 0–{}", input.len() - 1))?;

    let mut probe = Utils::add_noise(pattern, noise / 100.0, &mut StdRng::seed_from_u64(SEED));
    let (history, end) = train_bitmap(&bitmaps.patterns, oja).run(&mut probe, mode);
    let end = match end {
        TestEnd::Stable if history.last() == Some(pattern) => "Stable state: original recovered",
        TestEnd::Stable => "Stable state: not the original",
        TestEnd::Cycle => "Two-state cycle",
    };
    let frames = match mode {
        Mode::Sync => sync_frames(&history),
        Mode::Async => async_frames(&history, width, height),
    };

    let mut html = format!(
        r#"<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hopfield network steps</title>
<style>
  :root {{ color-scheme: light; font: 16px system-ui, sans-serif; color: #e5edf2; background: #10212b }}
  body {{ margin: 0 auto; max-width: 56rem; padding: 2rem 1rem }}
  h1 {{ margin: 0; font-size: 1.8rem }}
  p {{ color: #bfd0da }}
  .pair {{ display: flex; flex-wrap: wrap; gap: 1rem; margin: 1.5rem 0 1rem }}
  figure {{ flex: 1 1 14rem; margin: 0; padding: 1rem; background: #1a313e; border-radius: .75rem; text-align: center }}
  figcaption {{ margin-top: .5rem; color: #bfd0da }}
  canvas {{ display: block; margin: auto; width: 100%; height: auto; max-height: 58vh; object-fit: contain; image-rendering: pixelated }}
  .controls {{ display: flex; flex-wrap: wrap; gap: .5rem; margin: 1rem 0 }}
  button {{ padding: .6rem 1rem; border: 0; border-radius: .4rem; font: inherit; color: #10212b; background: #afe6d5; cursor: pointer }}
  button:disabled {{ opacity: .45; cursor: default }}
  input {{ width: 100%; margin: .4rem 0 0; accent-color: #afe6d5 }}
  output {{ display: block; margin-top: .4rem }}
</style>
<main>
  <h1>Hopfield network steps</h1>
  <p>{width} × {height} neurons · {label} · {update_mode} mode · {noise}% noise</p>
  <div class="pair">
    <figure><canvas id="bitmap" width="{width}" height="{height}" aria-label="Bitmap of the network state"></canvas><figcaption>Network state</figcaption></figure>
    <figure><canvas id="original" width="{width}" height="{height}" aria-label="Bitmap of the clean pattern"></canvas><figcaption>Clean pattern</figcaption></figure>
  </div>
  <label for="frame">Simulation frame</label>
  <input id="frame" type="range" min="0" value="0">
  <output id="status" for="frame" aria-live="polite"></output>
  <div class="controls">
    <button id="back">Back</button><button id="next">Next</button><button id="play">Play</button>
  </div>
</main>
<script>
"#
    );
    writeln!(html, "const original = \"{}\";", bits(pattern)).expect("writing to a String cannot fail");
    html.push_str("const states = [\n");
    for frame in &frames {
        writeln!(html, "\"{}\",", bits(&frame.state)).expect("writing to a String cannot fail");
    }
    html.push_str("];\nconst labels = [\n");
    for frame in &frames {
        let wrong = frame
            .state
            .iter()
            .zip(pattern)
            .filter(|(a, b)| a != b)
            .count();
        writeln!(html, "\"{} · {wrong} pixels differ from the clean pattern\",", frame.label)
            .expect("writing to a String cannot fail");
    }
    writeln!(html, "];\nconst result = '{end}';").expect("writing to a String cannot fail");
    html.push_str(include_str!("viz_2.js"));
    html.push_str("\n</script>\n</html>\n");
    fs::write(output, html).map_err(|e| format!("{}: {e}", output.display()))?;

    println!(
        "{end} · steps: {} · frames: {} · {}",
        history.len() - 1,
        frames.len(),
        output.display()
    );
    Ok(())
}

/// Sync: every step changes all neurons at once, so one step is one frame.
fn sync_frames(history: &[Vec<f32>]) -> Vec<Frame> {
    history
        .iter()
        .enumerate()
        .map(|(step, state)| Frame {
            state: state.clone(),
            label: if step == 0 { "input".to_owned() } else { format!("step {step}") },
        })
        .collect()
}

/// Async: neurons are updated in pixel order and each one once per pass, so the
/// state after the first `k` neurons of a pass equals the next state on those
/// neurons and the previous state on the rest. Frames are cut every few rows,
/// and chunks without any change are skipped.
fn async_frames(history: &[Vec<f32>], width: usize, height: usize) -> Vec<Frame> {
    let rows_per_frame = height.div_ceil(FRAMES_PER_PASS).max(1);
    let mut frames = vec![Frame {
        state: history[0].clone(),
        label: "input".to_owned(),
    }];
    for (pass, pair) in history.windows(2).enumerate() {
        let (before, after) = (&pair[0], &pair[1]);
        for first_row in (0..height).step_by(rows_per_frame) {
            let end_row = (first_row + rows_per_frame).min(height);
            let (start, split) = (first_row * width, end_row * width);
            if before[start..split] == after[start..split] {
                continue;
            }
            let mut state = after[..split].to_vec();
            state.extend_from_slice(&before[split..]);
            let rows = if end_row == first_row + 1 {
                format!("row {end_row}")
            } else {
                format!("rows {}–{end_row}", first_row + 1)
            };
            frames.push(Frame {
                state,
                label: format!("pass {} · {rows} updated", pass + 1),
            });
        }
    }
    frames
}

/// Pattern as a string of '1' (black) and '0' (white).
fn bits(state: &[f32]) -> String {
    state.iter().map(|&x| if x > 0.0 { '1' } else { '0' }).collect()
}