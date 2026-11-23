# Running the Hopfield project

Run these commands from the repository root.

```sh
cargo run --release
cargo run --release -- 2 6
cargo run --release -- 7
cargo run --release -- 1 2 3 4 5 6 7
```

With no arguments, the program runs experiments 1–6. Experiment 7 evaluates three 200×300 bitmaps using both learning rules and both update modes. You can pass several experiment numbers in one command. [Report in Polish](../RAPORT.md) · [Full results](results.txt)

## Visualization

```sh
cargo run --release -- viz src/data/large-25x25.csv 0 oja sync 25 target/recall.html
cargo run --release -- viz src/own/oscillation-train-5x5.csv 0 hebb sync 0 target/cycle.html src/own/oscillation-input-5x5.csv
cargo run --release -- viz src/large 0 hebb async 30 target/large.html
```

Open the generated HTML file in a browser. Use the slider and buttons to view the initial state, each simulation step, and the final result. In `async` mode, one step visits every pixel once and applies each change immediately.

The command takes `viz DATASET INDEX hebb|oja sync|async NOISE_PERCENT OUTPUT.html [INPUT]`. The zero-based index selects a pattern from the optional input or, if omitted, from the training dataset. The image dimensions determine the number of neurons.

A CSV file stores one pattern per row, with +1 for black and −1 for white. Include the dimensions in the filename, for example `patterns-10x15.csv`. You can also pass a plain PBM (P1) image or a directory of such images with matching dimensions. The program reads PBM files in alphabetical order.

## Step-by-step visualization

```sh
cargo run --release -- viz2 src/data/large-25x25.csv 0 oja async 30 target/steps.html
cargo run --release -- viz2 src/data/large-25x25.plus.csv 1 hebb async 20 target/steps-spurious.html
cargo run --release -- viz2 src/large 0 hebb async 30 target/steps-large.html
```

`viz2` takes the same arguments as `viz`. In `async` mode it splits every pass into about 25 frames, each covering a few rows of the image, and skips frames in which nothing changes. The slider then shows how the noise disappears row by row inside a pass. In `sync` mode all neurons change at once, so each step is one frame, as in `viz`. The page shows the clean pattern next to the network state, and every frame reports how many pixels still differ from it. The program prints the number of steps and frames after writing the file.