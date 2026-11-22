# Pamięć skojarzeniowa z siecią Hopfielda

## Model i reguły uczenia

Stan s zawiera n pikseli o wartościach −1 i +1. Wagi W są symetryczne, a wᵢᵢ = 0. Neuron oblicza pole lokalne hᵢ = Σⱼ wᵢⱼsⱼ i przyjmuje jego znak. Gdy pole wynosi zero, zachowuje swój stan. Wariant grupowy Oji traktuje również |hᵢ| ≤ 10⁻¹⁰ jako remis. Wzorzec x jest punktem stałym, gdy w modelu dokładnym wszystkie marginesy xᵢhᵢ są nieujemne. Ściśle dodatni margines oznacza, że neuron ma dodatnie pole w kierunku swojego obecnego stanu.

W trybie asynchronicznym jeden krok obejmuje pełne przejście po neuronach. Po każdej zmianie następny neuron korzysta z aktualnego stanu. Energia E(s) = −½sᵀWs zmienia się przy aktualizacji neuronu i o −(sᵢ′−sᵢ)hᵢ. Każda zmiana znaku obniża energię. Ponieważ zbiór stanów jest skończony, sieć dochodzi do punktu stałego. Gdy wszystkie aktualizacje korzystają z poprzedniego stanu (tryb synchroniczny), ten argument o energii nie ma zastosowania. Symetryczna dynamika progowa może osiągnąć punkt stały lub cykl dwustanowy [1, 5].

**Hebb.** Dla i ≠ j stosujemy wᵢⱼ = Σₚ xᵢᵖxⱼᵖ/n, gdzie p przebiega po P wzorcach. Wspólny znak pikseli wzmacnia połączenie dodatnie, a przeciwny znak wzmacnia połączenie ujemne. Równoważnie, aktywacja neuronów zależy od sumy iloczynów skalarnych wzorców z bieżącym stanem. Duży dodatni iloczyn skalarny sprawia, że dany wzorzec silnie wpływa na następny stan. Zapis energii

```text
E(s) = − Σₚ (xᵖ·s)² / (2n) + P/2
```

pokazuje, dlaczego w pobliżu zapamiętanych wzorców mogą powstawać minima. Kwadrat daje także minima przy ich negatywach i pozwala na stany mieszane. Dla wzorca q margines składa się z sygnału własnego 1 − P/n i przesłuchu Σₚ≠q xᵢᵖxᵢᑫ(xᵖ·xᑫ)/n. Gdy skorelowane wzorce wzmacniają się na tle i mają odmienne piksele w detalu, przesłuch może zmienić znak pola właśnie w detalu. Granica obciążenia około 0,138 dotyczy klasycznego modelu z losowymi, słabo skorelowanymi wzorcami oraz odzyskiwania stanów silnie z nimi skorelowanych [1, 2].

**Oja.** Reguła jednego neuronu Δw = ηy(x − yw), gdzie y = wᵀx, dodaje wkład Hebba ηyx i korektę −ηy²w. Korekta tłumi nieograniczony wzrost normy. Średnia aktualizacja ma postać η(Cw − (wᵀCw)w) dla C = E[xxᵀ]. Po znormalizowaniu wzmocnienie rośnie w kierunku składowej o największej wartości własnej. W wersji wielowyjściowej korekta dotyczy części wejścia wyjaśnionej wszystkimi wyjściami [3, 4].

W projekcie stosujemy regułę podprzestrzenną dla macierzy W zapisywanej wierszami. Obliczamy y = Wx, v = Wᵀy, a następnie ΔW = ηy(x − v)ᵀ. Punkt wyjścia to wagi Hebba podzielone przez największą sumę modułów w wierszu. Przekątna pozostaje zerowa, a po każdej epoce uśredniamy W i Wᵀ. Użyto η = 0,5/n, do 300 epok i progu maksymalnej aktualizacji 10⁻⁶. Mechanizm korekty widać w postaci średniej aktualizacji przed narzuceniem ograniczeń, ηWC(I − WᵀW). Dla wspólnego kierunku własnego symetrycznych W i C przyrost ma czynnik λw(1 − w²), więc dobrze reprezentowany kierunek przestaje dominować przez sam wzrost swojej wagi. To wyjaśnia, czemu reguła może lepiej zachować mniej dominujące cechy. Po wyzerowaniu przekątnej, symetryzacji i skończonej liczbie epok znaki pól trzeba sprawdzić na każdym wzorcu.

**Duże obrazy.** Wizualizacja i eksperyment bitmap używają dotychczasowej macierzy sieci dla obrazów do 2048 pikseli. Powyżej tej granicy Hebb korzysta z istniejącego `LargeHopfield`, który wylicza pola z iloczynów skalarnych wzorców z bieżącym stanem. Wspólny dodatni czynnik 1/n można pominąć, bo decyzja zależy od znaku pola. Oja korzysta z macierzy g×g dla g grup neuronów. Neurony o identycznych sygnaturach uczących mają te same wagi, a sygnatury przeciwne łączymy po zmianie znaku. Nadal odejmujemy wkład neuronu do samego siebie. Przy trzech wzorcach powstają najwyżej cztery grupy, podczas gdy pełna macierz `f64` dla 60 000 neuronów zajęłaby 28,8 GB. Przy wielu różnych sygnaturach macierz grup może nadal mieć rozmiar n×n. Zgrupowane uczenie i pojedyncze kroki porównano z pełną macierzą w teście jednostkowym.

## Metodyka

Odwracaliśmy losowo ustaloną liczbę różnych pikseli (0, 10, 20, 30, 40, 50%), po 10 prób na wzorzec i poziom. W eksperymentach 2–5 program bazowy używa kolejnych realizacji generatora dla każdej konfiguracji. W eksperymencie 7 wszystkie konfiguracje dostają te same realizacje uszkodzenia. Pomiar dokładnego odzyskania wymagał stabilnego wyniku identycznego z oryginałem. Udział zgodnych pikseli liczono również przy nieudanym odzyskaniu i w stanie kończącym ślad cyklu. Liczność badano na kolejnych prefiksach zbioru. Osobny eksperyment obejmował po 100 niezależnych wejść z pikselami losowanymi z jednakowym prawdopodobieństwem. Generator inicjowano ziarnem 42. [Pełne wyniki](docs/wyniki.txt) można odtworzyć poleceniem `cargo run --release -- 1 2 3 4 5 6 7`.

## 1. Stabilność zbiorów dostarczonych na zajęcia

Poniżej średnia bezwzględna wartość znormalizowanego iloczynu skalarnego między różnymi wzorcami i liczba punktów stałych.

| Zbiór | Średni overlap | Hebb | Oja |
| --- | ---: | ---: | ---: |
| OCRA-12x30-cut | 0,272 | 1/7 | 7/7 |
| animals-14x9 | 0,212 | 2/6 | 6/6 |
| large-25x25 | 0,142 | 5/6 | 6/6 |
| large-25x25.plus | 0,171 | 1/15 | 15/15 |
| large-25x50 | 0,567 | 0/6 | 6/6 |
| letters-14x20 | 0,728 | 0/26 | 0/26 |
| letters-abc-8x12 | 0,806 | 0/3 | 3/3 |
| small-7x7 | 0,286 | 3/5 | 5/5 |

Łącznie Hebb utrzymał 12 z 74 wzorców, a Oja 48. Oja utrzymała komplet w siedmiu zbiorach. W literach 14×20 46,4% pozycji ma wspólny kolor we wszystkich 26 wzorcach. Średni udział czerni wynosi 14,4%, a niektóre pary mają overlap bliski 0,99. Hebb zmienił w pierwszym kroku od 10 do 46 pikseli każdego oryginału, a Oja od 7 do 65. W tym przypadku struktura skorelowanego tła i liczność powodują duże zakłócenie słabszych cech rozróżniających. W zestawie ABC 8×12 podobieństwo jest jeszcze wyższe, ale korekta Oji wystarczyła dla trzech wzorców. Ponieważ obciążenie wszystkich zestawów pozostaje poniżej 0,138, sama liczba neuronów na wzorzec nie przewiduje tych wyników [1, 2, 4].

## 2. Wspólny blok w zestawie 25×50

Wszystkie sześć górnych połówek jest piksel w piksel identycznych z odpowiadającymi wzorcami 25×25. Dolne połówki są w całości czarne. Dla korelacji ze znakiem o = (x·y)/n wspólny blok daje dokładnie o₍₂₅ₓ₅₀₎ = (o₍₂₅ₓ₂₅₎ + 1)/2. W zestawie kontrolnym dolne połówki wygenerowano niezależnie dla każdego wzorca.

| Wariant | Średni overlap | Hebb stabilne | Hebb odzysk sync/async, szum 20% | Oja stabilne | Oja odzysk sync/async, szum 20% |
| --- | ---: | ---: | ---: | ---: | ---: |
| 25×25 | 0,142 | 5/6 | 67% / 50% | 6/6 | 100% / 100% |
| 25×50, wspólny blok | 0,567 | 0/6 | 0% / 0% | 6/6 | 100% / 100% |
| 25×50, losowy blok | 0,076 | 6/6 | 100% / 100% | 6/6 | 100% / 100% |

Wspólny blok podnosi pola od innych wzorców w górnej części. Gdy te pola głosują przeciw cechom oryginału, rozpoznanie kończy się podobną bitmapą (około 87% zgodnych pikseli w teście 3 przy 20% szumu dla Hebba), ale szczegóły ulegają zmianie. Próba kontrolna pokazuje, że przy tym samym rozmiarze i liczności wyniki wracają po rozproszeniu korelacji. Dla Oji wpływ jest niewielki w tym teście. Przy 40% szumu i aktualizacji asynchronicznej Oja odzyskała 98% obrazów 25×25 oraz 100% w obu wariantach 25×50.

## 3. Liczność, rodzaj zbioru i uszkodzenie

W `large-25x25.plus`, przy 20% szumu i aktualizacji asynchronicznej, Hebb odzyskał 100% oryginałów dla pierwszych czterech wzorców, 94% dla pięciu, 53% dla sześciu i 7% dla piętnastu. Oja uzyskała 100% dla wszystkich prefiksów. W literach 14×20 Hebb osiągnął 100% dla dwóch wzorców i 0% od trzech, a Oja 100% do czterech i 0% we wszystkich zbadanych prefiksach od szesnastu. Kolejne prefiksy zmieniają jednocześnie liczność i zestaw korelacji, dlatego spadek nie musi być jednostajny.

Dla pełnego zbioru `small-7x7` i Oji w trybie synchronicznym odzysk wyniósł 100% przy 20% szumu, 74% przy 30% i 18% przy 40%. Dla `large-25x25.plus` te trzy wyniki wyniosły po 100%. Na literach 14×20 dokładny odzysk wynosił 0% w obu regułach na wszystkich poziomach szumu, mimo około 90% zgodności pikseli przy szumie 20%.

Losowe odwrócenie 20% pikseli porównano z zakryciem białym pasem 20% powierzchni (wysokość zaokrąglona do pełnych wierszy). Dla Hebba w trybie asynchronicznym pas obniżył odzysk na 25×25 z 43% do 33%, a w zbiorze zwierząt z 12% do 7%. Biały pas zachowuje istniejące białe piksele, ale usuwa skorelowane przestrzennie fragmenty figury. Te protokoły badają dwa różne uszkodzenia.

## 4. Grupy wzorców podobnych

Porównano po dwa podzbiory o równej liczności z `large-25x25.plus.csv`. Pierwszy wybierano zachłannie, minimalizując podobieństwo do już dobranych wzorców. Drugi budowano, dobierając rozłączne pary o najwyższym podobieństwie.

| Liczba | Zbiór | Overlap | Hebb stabilne | Hebb odzysk przy 20% szumu | Oja odzysk przy 20% szumu |
| ---: | --- | ---: | ---: | ---: | ---: |
| 6 | zróżnicowany | 0,074 | 6/6 | 100% | 100% |
| 6 | z grupami | 0,358 | 0/6 | 0% | 100% |
| 8 | zróżnicowany | 0,093 | 8/8 | 90% | 100% |
| 8 | z grupami | 0,292 | 0/8 | 0% | 100% |

Dla Hebba wszystkie 60 i 80 asynchronicznych prób na grupowanych podzbiorach zakończyły się stanami pasożytniczymi. Oja zachowała wszystkie wzorce w czterech podzbiorach. Przy równej liczności i wymiarach różnicę wyjaśnia tu przesłuch wzorców.

## 5. Losowe wejście

Na obrazach zwierząt Oja asynchroniczna dała 42 oryginały, 37 negatywów i 21 stanów pasożytniczych. Hebb asynchroniczny dał 2 oryginały, 1 negatyw i 97 stanów pasożytniczych. Na zestawie liter 14×20 Hebb kończył w dwóch stanach pasożytniczych, a Oja synchroniczna w 99 na 100 prób osiągała cykl dwustanowy. Dla Oji asynchronicznej każda z tych prób zakończyła się w punkcie stałym pasożytniczym. Losowy wektor trafia do basenu tego atraktora, w którego obszarze leży na wejściu. Negatywy pojawiają się naturalnie, bo przy braku progów zmiana wszystkich znaków odwraca następny stan i nie zmienia energii.

## 6. Własne wzorce 5×5

**Zbiór w pliku.** `src/own/stable-5x5.csv` zawiera 19 prawie ortogonalnych wzorców opartych na macierzy Hadamarda. Hebb zachował wszystkie 19 z minimalnym marginesem 2/25, a Oja zachowała 10/19.

**Konstrukcja o dużej liczności.** We wszystkich kombinacjach pierwszych 22 pozycji wybieramy dokładnie 11 wartości +1. Na ostatnich trzech pozycjach wpisujemy ten sam znak, niezależnie od pierwszej części. Otrzymujemy M = 2·C(22,11) = **1 410 864 różnych wzorców**. Dla różnych i, j w pierwszym bloku średni iloczyn xᵢxⱼ wynosi −1/21, bo suma 22 pikseli jest zerowa. Wagi pomiędzy blokami są zerowe, bo oba znaki trójki są reprezentowane po równo. W ostatniej trójce średni iloczyn wynosi 1. Po nauczeniu Hebba margines pierwszych 22 pikseli wynosi M/(21·25), a margines każdego z ostatnich trzech to 2M/25. Oba są dodatnie dla każdego wzorca. Program zlicza rodzinę i wyznacza marginesy ze wzoru bez materializacji miliona obrazów. Jest to dolne ograniczenie na możliwą liczność zbioru ściśle stabilnego.

Jeśli dopuszczamy remis na części pikseli, można wziąć wszystkie wektory z dokładnie 12 lub 13 dodatnimi pikselami. Symetria daje ujemne wagi o tej samej wartości dla każdej pary. Przy sumie s równej ±1 pole większościowych pikseli wynosi zero, a pole mniejszościowych ma poprawny znak. Dla przyjętej reguły remisu stabilne jest wówczas 2·C(25,12) = **10 400 600** wzorców.

Przy regule zachowania znaku w remisie formalne maksimum to wszystkie 2²⁵ = **33 554 432** bitmapy. Dla każdej pary różnych pikseli dokładnie połowa wzorców daje iloczyn +1, a połowa −1. Uczenie Hebba daje więc W = 0. Każde wejście pozostaje na miejscu, także po uszkodzeniu. Ten skrajny przypadek pokazuje, dlaczego do oceny pamięci obok liczby punktów stałych mierzymy odzyskiwanie oryginałów z szumu.

**Cykl.** Uczymy Hebba na krzyżu i znaku X z `src/own/oscillation-train-5x5.csv`, po czym podajemy ramkę z `src/own/oscillation-input-5x5.csv`. W trybie synchronicznym wszystkie piksele odczytują ten sam poprzedni stan. Poniżej czarny piksel oznaczono przez `#`.

```text
wejście    krok 1    krok 2
#####      .#.#.     #####
#...#      #####     #...#
#...#      .#.#.     #...#
#...#      #####     #...#
#####      .#.#.     #####
```

Powrót do wejścia po dwóch krokach zamyka cykl. W trybie asynchronicznym wejście przechodzi po jednym przebiegu ze zmianami w negatyw X.

## 7. Bitmapy 200×300

Wybrano fotografie `camera`, `coins` i `moon` z scikit-image 0.24.0. Przeskalowano je do 200×300 z centralnym kadrem filtrem Lanczos, po konwersji do skali szarości. Piksele o jasności mniejszej od całkowitej części mediany (152, 89 i 112) stały się czarne. Pliki PBM i informacje o źródłach są w [src/large](src/large/). Średni overlap wyniósł 0,159. Obie reguły utrzymały 3/3 wzorców.

| Reguła i tryb | Dokładne, 0–40% szumu | Dokładne, 50% szumu | Zgodne piksele, 50% szumu |
| --- | ---: | ---: | ---: |
| Hebb sync | 30/30 na poziom | 0/30 | 52% |
| Hebb async | 30/30 na poziom | 0/30 | 61% |
| Oja sync | 30/30 na poziom | 0/30 | 53% |
| Oja async | 30/30 na poziom | 0/30 | 61% |

Dla uszkodzenia z odwróceniem dokładnie k pikseli overlap wejścia z oryginałem wynosi 1 − 2k/n. Przy 40% pozostaje sygnał 0,2, a przy 50% wejście jest z definicji ortogonalne do oryginału. Wynik dotyczy trzech zrównoważonych po binaryzacji zdjęć przy bardzo małym obciążeniu P/n = 0,00005.

## Literatura

1. J. J. Hopfield (1982), *Neural networks and physical systems with emergent collective computational abilities*. [DOI](https://doi.org/10.1073/pnas.79.8.2554)
2. D. J. Amit, H. Gutfreund, H. Sompolinsky (1985), *Storing Infinite Numbers of Patterns in a Spin-Glass Model of Neural Networks*. [DOI](https://doi.org/10.1103/PhysRevLett.55.1530)
3. E. Oja (1982), *A simplified neuron model as a principal component analyzer*. [DOI](https://doi.org/10.1007/BF00275687)
4. E. Oja (1989), *Neural Networks, Principal Components, and Subspaces*. [DOI](https://doi.org/10.1142/S0129065789000475)
5. E. Goles, J. Olivos (1980), *Periodic behaviour of generalized threshold functions*. [DOI](https://doi.org/10.1016/0012-365X(80)90121-1)
