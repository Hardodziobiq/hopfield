# Pamięć skojarzeniowa Hopfielda

Implementacja w Rust z regułami Hebba i Oji, aktualizacją synchroniczną i asynchroniczną oraz wizualizacją kolejnych stanów. [Raport](RAPORT.md) zawiera opis metod, wyniki i interpretację doświadczeń.

## Uruchomienie

Wymagany jest Rust z obsługą edycji 2024. Polecenia uruchamiaj w głównym katalogu repozytorium.

```sh
cargo test
cargo run --release
cargo run --release -- 2 6 7
```

| Nr | Eksperyment |
| --- | --- |
| 1 | Stabilność wzorców dostarczonych na zajęcia |
| 2 | Porównanie wersji 25×25 i 25×50 wraz z próbą kontrolną |
| 3 | Odzyskiwanie przy różnej liczbie wzorców i różnych uszkodzeniach |
| 4 | Zbiory z podobnymi grupami i zbiory zróżnicowane |
| 5 | Losowe wejścia, negatywy, stany pasożytnicze i cykle |
| 6 | Zbiory własne 5×5 i przykład cyklu dwustanowego |
| 7 | Trzy bitmapy 200×300 |

## Wizualizacja kroków

```sh
cargo run --release -- viz src/data/large-25x25.csv 0 oja sync 25 target/pamiec.html
cargo run --release -- viz src/own/oscillation-train-5x5.csv 0 hebb sync 0 target/cykl.html src/own/oscillation-input-5x5.csv
cargo run --release -- viz src/large 0 hebb async 30 target/duza.html
```

Otwórz wygenerowany plik HTML w przeglądarce. Suwak i przyciski pozwalają obejrzeć stan początkowy, kolejne bitmapy i wynik. Krok asynchroniczny jest jednym przejściem po neuronach w kolejności pikseli. Zmiany w tym samym przejściu od razu wpływają na następne neurony.

Składnia polecenia to `viz ZBIÓR INDEKS hebb|oja sync|async SZUM_PROCENT WYNIK.html [WEJŚCIE]`. Domyślnie wzorzec wejściowy wybiera się ze zbioru uczącego. Jeśli podasz dodatkowy plik, indeks dotyczy tego pliku. Indeksy zaczynają się od 0, a szum to odsetek losowo odwróconych pikseli.

Liczbę neuronów ustalają wymiary wejściowej bitmapy. Dla CSV wpisz `szerokośćxwysokość` w nazwę pliku, np. `wzorce-10x15.csv`. Każdy wiersz zawiera wzorzec zapisany od lewej do prawej i od góry do dołu, z wartościami `1` dla czerni i `-1` dla bieli. Można też podać obraz PBM w formacie tekstowym P1 lub katalog takich obrazów o wspólnych wymiarach. Katalog jest wczytywany w kolejności alfabetycznej.
