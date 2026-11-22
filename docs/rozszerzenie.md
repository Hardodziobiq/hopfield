# Rozszerzenie projektu Hopfielda

Polecenia uruchamiaj z głównego katalogu repozytorium.

```sh
cargo run --release
cargo run --release -- 2 6
cargo run --release -- 7
cargo run --release -- 1 2 3 4 5 6 7
```

Bez argumentów program uruchamia dotychczasowe eksperymenty 1–6. Polecenie z numerem 7 bada trzy bitmapy 200×300 obiema regułami w obu trybach. Możesz łączyć numery w jednym wywołaniu. [Raport](../RAPORT.md) opisuje doświadczenia, a [zapis pomiarów](wyniki.txt) zawiera ich pełne wyniki.

## Wizualizacja

```sh
cargo run --release -- viz src/data/large-25x25.csv 0 oja sync 25 target/pamiec.html
cargo run --release -- viz src/own/oscillation-train-5x5.csv 0 hebb sync 0 target/cykl.html src/own/oscillation-input-5x5.csv
cargo run --release -- viz src/large 0 hebb async 30 target/duza.html
```

Wygenerowany plik HTML otwórz w przeglądarce. Suwak i przyciski pokazują stan początkowy, kolejne kroki i wynik. Krok w trybie `async` to jedno przejście przez wszystkie piksele z natychmiastowym zastosowaniem każdej zmiany.

Argumenty polecenia to `viz ZBIÓR INDEKS hebb|oja sync|async SZUM_PROCENT WYNIK.html [WEJŚCIE]`. Indeks zaczyna się od zera i wybiera wzorzec z pliku wejściowego lub ze zbioru uczącego, jeśli dodatkowego pliku brak. Podane wymiary określają liczbę neuronów. CSV przechowuje po jednym wzorcu w wierszu, w notacji +1 dla czerni i −1 dla bieli. Jego nazwa zawiera wymiary, jak w `wzorce-10x15.csv`. Można też wskazać tekstowy obraz PBM P1 albo katalog takich obrazów o zgodnych wymiarach. Program czyta pliki katalogu w kolejności alfabetycznej.
