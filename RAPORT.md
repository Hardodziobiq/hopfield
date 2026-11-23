<div align="center">
  <h1>Sieci neuronowe</h1>

  <img src="docs/logo-mini.jpg" width="112" height="112" alt="Logo Wydziału Matematyki i Nauk Informacyjnych">
  <p><strong>Politechnika Warszawska</strong><br>Wydział Matematyki i Nauk Informacyjnych<br>24.11.2026</p>
  <h1>Pamięć skojarzeniowa z użyciem sieci Hopfielda</h1>
  <p><strong>Autorzy</strong><br>Maksymilian Majewski<br>Łukasz Przybylski</p>
  <p><strong>Prowadzący</strong><br>mgr inż. Mikołaj Małkiński</p>
</div>

<div style="break-after: page; page-break-after: always;"></div>

## Oznaczenia

| Symbol lub nazwa w kodzie | Znaczenie |
| --- | --- |
| $n$, $i,j,k$ | Liczba neuronów i ich indeksy. Jeden piksel to jeden neuron, więc $n = \text{width}\cdot\text{height}$. |
| $\mathbf a^{(t)}$ | Stan sieci w chwili $t$ (`probe`). Każdy element $a_i^{(t)}$ to $+1$ dla czerni lub $-1$ dla bieli. |
| $\tilde{\mathbf x}$ | Wzorzec wejściowy, na przykład obraz po dodaniu szumu. Od niego zaczyna się symulacja, $\mathbf a^{(0)}=\tilde{\mathbf x}$. |
| $\mathbf W=[w_{ij}]$, $\mathbf b$ | Macierz wag (`weights`) oraz wektor pobudzeń zewnętrznych (ang. *biases*). W projekcie $\mathbf b=\mathbf 0$, $w_{ij}=w_{ji}$ i $w_{ii}=0$. |
| $z_i$ | Preaktywacja, czyli pole lokalne neuronu $i$. |
| $\mathbf x^{(p)}$, $N$, $\mathbf X$ | Wzorzec uczący $p$, liczba wzorców i macierz $\mathbf X=[\mathbf x^{(1)},\ldots,\mathbf x^{(N)}]$. Każdy wzorzec to jedna kolumna. |
| $\mathbf y$, $\mathbf v$, $\eta$ | Odpowiedź liniowa, rekonstrukcja wzorca z odpowiedzi i krok uczenia Oji. |
| $m_p$, $\alpha$ | Znormalizowany iloczyn skalarny stanu i wzorca $p$ oraz obciążenie pamięci $\alpha=N/n$. |
| $E(\mathbf a)$ | Energia stanu $\mathbf a$. |

## Cel projektu

Sieć Hopfielda ma odtworzyć zapamiętaną bitmapę na podstawie uszkodzonego wejścia. Porównaliśmy uczenie regułą Hebba i regułą Oji oraz dwa tryby aktualizacji neuronów.

**Główny wynik.** Na ośmiu zbiorach po uczeniu regułą Hebba stabilnych było 12 z 74 wzorców, a po uczeniu regułą Oji 48 z 74. Na trzech bitmapach 200×300 obie metody odtworzyły oryginały we wszystkich próbach przy szumie do 40%.

## 1. Aktualizacja neuronów i wizualizacja

Neuron sumuje ważone stany i pobudzenie zewnętrzne. Znak tej sumy wyznacza nowy stan. Przy zerowej preaktywacji neuron zachowuje dotychczasowy stan. Wariant grupowy Oji stosuje tę samą zasadę, gdy moduł preaktywacji nie przekracza $10^{-10}$.

$$
z_i^{(t+1)} = \sum_{j=1}^{n} w_{ij} a_j^{(t)} + b_i,
\qquad
a_i^{(t+1)} =
\begin{cases}
+1, & z_i^{(t+1)}>0,\\
a_i^{(t)}, & z_i^{(t+1)}=0,\\
-1, & z_i^{(t+1)}<0.
\end{cases}
$$

| Tryb | Aktualizacja | Warunek zakończenia |
| --- | --- | --- |
| Synchroniczny (`sync`) | Wszystkie neurony korzystają z tego samego stanu z początku kroku. | Punkt stały lub cykl dwustanowy. |
| Asynchroniczny (`async`) | W pojedynczej aktualizacji zmienia się najwyżej jeden neuron $k$. Następny neuron widzi już tę zmianę. | Pełne przejście bez zmian. |

**Wizualizacja.** Wymiary bitmapy wyznaczają liczbę neuronów. Program zapisuje stan wejściowy i kolejne stany sieci do pliku HTML z suwakiem. W trybie asynchronicznym jedna klatka odpowiada pełnemu przejściu po neuronach.

**Zbieżność.** Przy symetrycznych wagach i zerowej przekątnej każda rzeczywista asynchroniczna zmiana neuronu $k$ zmniejsza energię. Stanów jest skończenie wiele, więc sieć osiąga punkt stały. Aktualizacja synchroniczna może też zakończyć się cyklem długości 2. [1, 5, 6, slajdy 19 i 27–28]

$$
E(\mathbf a) = -\frac12 \mathbf a^\top\mathbf W\mathbf a - \mathbf b^\top\mathbf a,
\qquad
\Delta E = -\Delta a_k\,z_k,
\qquad
\Delta a_k = a_k^{(t+1)}-a_k^{(t)}.
$$

## 2. Reguły uczenia

### Reguła Hebba

**Wzór w implementacji.** Funkcja `weights_hebb` sumuje wkłady wszystkich wzorców uczących. Symbol $\mathbf I$ oznacza macierz jednostkową. [6, slajd 13]

$$
w_{ij} =
\begin{cases}
\displaystyle \frac{1}{n}\sum_{p=1}^{N} x_i^{(p)}x_j^{(p)}, & i\ne j,\\
0, & i=j,
\end{cases}
\qquad
\mathbf W=\frac{1}{n}\left(\mathbf X\mathbf X^\top-N\mathbf I\right).
$$

**Rozpoznawanie.** Każdy wzorzec wpływa na znak pola lokalnego proporcjonalnie do swojego podobieństwa $m_p$ do bieżącego stanu. Jeśli wejście jest bliskie oryginałowi, jego wkład może skorygować błędne piksele. [6, slajd 16]

$$
m_p=\frac{1}{n}\bigl(\mathbf x^{(p)}\bigr)^\top\mathbf a,
\qquad
z_i=\sum_{p=1}^{N}m_p x_i^{(p)}-\frac{N}{n}a_i.
$$

**Sygnał i przesłuch.** Dla $\mathbf a=\mathbf x^{(q)}$ pierwszy składnik podtrzymuje znak piksela oryginału, a $C_i^{(q)}$ opisuje wpływ pozostałych wzorców. Piksel zachowuje stan, gdy $x_i^{(q)}z_i\geq 0$. Przy dodatnim iloczynie jest ściśle stabilny. [6, slajdy 18 i 22]

$$
\begin{aligned}
z_i &= \underbrace{\frac{n-1}{n}x_i^{(q)}}_{\text{sygnał}}
       + \underbrace{C_i^{(q)}}_{\text{przesłuch}},\\
C_i^{(q)} &= \frac1n\sum_{p\ne q} x_i^{(p)}
                      \sum_{j\ne i} x_j^{(p)}x_j^{(q)}.
\end{aligned}
$$

### Reguła Oji

Dla jednego neuronu z wektorem wag $\mathbf w$ i odpowiedzią $y=\mathbf w^\top\mathbf x$ reguła Oji dodaje do składnika Hebba korektę, która rośnie wraz z kwadratem odpowiedzi. [3]

$$
\Delta\mathbf w = \eta y\mathbf x - \eta y^2\mathbf w.
$$

**Kolejność w kodzie.** Funkcja `weights_oja` zaczyna od macierzy Hebba. Dzieli ją przez największą sumę modułów wag w wierszu, jeśli ta suma jest dodatnia. Następnie dla kolejnych wzorców uczących $\mathbf x=\mathbf x^{(p)}$ stosuje wariant wielowyjściowy [4].

$$
\begin{aligned}
\mathbf y &= \mathbf W\mathbf x
   && \text{odpowiedź na wzorzec},\\
\mathbf v &= \mathbf W^\top\mathbf y
   && \text{rekonstrukcja},\\
w_{ij} &\leftarrow w_{ij}+\eta\,y_i(x_j-v_j)
   && \text{poprawka dla } i\ne j,\qquad \eta=\frac{0{,}5}{n}.
\end{aligned}
$$

Po każdej epoce program symetryzuje macierz, $\mathbf W\leftarrow(\mathbf W+\mathbf W^\top)/2$, i sprawdza największy moduł poprawki. Uczenie kończy się, gdy ta wartość spadnie poniżej $10^{-6}$ lub gdy minie 300 epok.

**Dlaczego to pomaga?** Jeśli $\mathbf v$ dobrze odtwarza wspólne tło obrazów, różnica $\mathbf x-\mathbf v$ jest mała na jego pikselach. Kolejne aktualizacje słabiej wzmacniają tło, a piksele odróżniające wzorce nadal wpływają na wagi.

### Obliczenia dla dużych bitmap

W wizualizacji i eksperymencie 7 dla obrazów powyżej 2048 pikseli program wybiera reprezentację zależnie od reguły uczenia.

- **Hebb (`LargeHopfield`)** przechowuje wzorce i oblicza iloczyny skalarne $u_p$, gdzie $u_p=n m_p$. Przy ustalaniu znaku pola program pomija wspólny dodatni czynnik $1/n$. Po aktualizacji asynchronicznej neuronu $k$ zwiększa $u_p$ o $x_k^{(p)}\Delta a_k$.
- **Oja (`GroupedHopfield`)** przypisuje neurony do grup według ich znaków w kolejnych wzorcach. Sygnatury przeciwne trafiają do tej samej grupy z przeciwnym znakiem odniesienia. Każdy piksel zachowuje własny stan. Przy trzech wzorcach powstają najwyżej cztery grupy.

$$
u_p=\bigl(\mathbf x^{(p)}\bigr)^\top\mathbf a,
\qquad
z_i=\frac{1}{n}\left(\sum_{p=1}^{N}x_i^{(p)}u_p-Na_i\right).
$$

## 3. Jak mierzyliśmy skuteczność

| Metryka | Warunek lub sposób liczenia |
| --- | --- |
| Stabilność wzorca | Podajemy oryginał bez szumu. Jeśli pierwszy krok synchroniczny nie zmienia żadnego piksela, kolejne kroki także pozostawią ten stan bez zmian. |
| Dokładny odzysk | Podajemy zaszumioną kopię i sprawdzamy, czy sieć zatrzymuje się na obrazie identycznym z oryginałem. |
| Zgodność pikseli | Odsetek pikseli równych oryginałowi w ostatniej zapisanej klatce, także po nieudanym odzysku. |

W głównych próbach losowo odwracaliśmy 0, 10, 20, 30, 40 lub 50% pikseli. Dla każdego wzorca i poziomu wykonaliśmy po 10 prób. Ziarno generatora wynosiło 42. W tabelach średnie podobieństwo par to średnia wartość bezwzględna $m_p$ dla różnych par wzorców.

## 4. Eksperymenty

### 4.1. Stabilność dostarczonych wzorców

**Metoda.** Dla każdego zbioru uczyliśmy obie sieci na komplecie wzorców i sprawdzaliśmy każdy oryginał bez szumu.

| Zbiór | Średnie podobieństwo par | Stabilne po Hebbie | Stabilne po Oji |
| --- | ---: | ---: | ---: |
| OCRA-12x30-cut | 0,272 | 1/7 | 7/7 |
| animals-14x9 | 0,212 | 2/6 | 6/6 |
| large-25x25 | 0,142 | 5/6 | 6/6 |
| large-25x25.plus | 0,171 | 1/15 | 15/15 |
| large-25x50 | 0,567 | 0/6 | 6/6 |
| letters-14x20 | 0,728 | 0/26 | 0/26 |
| letters-abc-8x12 | 0,806 | 0/3 | 3/3 |
| small-7x7 | 0,286 | 3/5 | 5/5 |

**Interpretacja.** Reguła Oji zachowała komplet wzorców w siedmiu zbiorach. W zbiorze 26 liter 46,4% pozycji ma wspólny kolor. Wspólne tło zwiększa przesłuch, który może odwrócić znaki pikseli odróżniających litery [2, 6, slajd 22]. W zbiorze ABC reguła Oji zachowała wszystkie trzy obrazy mimo wyższego średniego podobieństwa. Na stabilność wpływają zarówno korelacje, jak i liczba wzorców.

### 4.2. Wzorce 25×25 i 25×50 ze wspólnym blokiem

**Metoda.** Górne połówki obrazów 25×50 odpowiadają obrazom 25×25, a dolne są czarne. W próbie kontrolnej wypełniliśmy dolną połowę każdego wzorca niezależnie losowanymi pikselami. Wspólny czarny blok dodaje do iloczynu skalarnego 625, a łączna liczba pikseli rośnie z 625 do 1250.

$$
m_p^{(25\times 50)} = \frac{m_p^{(25\times 25)}+1}{2}.
$$

| Wariant | Średnie podobieństwo par | Hebb stabilne | Hebb odzysk sync / async | Oja stabilne | Oja odzysk sync / async |
| --- | ---: | ---: | ---: | ---: | ---: |
| 25×25 | 0,142 | 5/6 | 67% / 50% | 6/6 | 100% / 100% |
| 25×50, wspólny blok | 0,567 | 0/6 | 0% / 0% | 6/6 | 100% / 100% |
| 25×50, losowy blok | 0,076 | 6/6 | 100% / 100% | 6/6 | 100% / 100% |

Odzysk zmierzyliśmy przy 20% szumu. Wspólny blok zwiększył podobieństwo wzorców, a przez to przesłuch także w górnej połowie obrazu. Losowy blok przywrócił skuteczność reguły Hebba przy tej samej liczbie neuronów i wzorców.

### 4.3. Liczność zbioru, poziom szumu i rodzaj uszkodzeń

| Porównanie | Wynik dokładnego odzysku |
| --- | --- |
| Pierwsze 4, 5, 6 i 15 wzorców z `large-25x25.plus`, 20% szumu, async | Hebb 100%, 94%, 53% i 7%. Oja 100% dla wszystkich prefiksów. |
| Prefiksy liter 14×20, 20% szumu, async | Hebb 100% dla dwóch wzorców i 0% od trzech. Oja 100% do czterech i 0% od szesnastu. |
| Oja sync przy 20%, 30% i 40% szumu | `small-7x7` osiągnął 100%, 74%, 18%. `large-25x25.plus` osiągnął 100% przy każdym z tych poziomów. |
| Losowy szum 20% i biały pas około 20% powierzchni, Hebb async | Dla 25×25 wynik spadł z 43% do 33%, a dla zwierząt z 12% do 7%. |

**Interpretacja.** Na pełnym zbiorze liter 14×20 żadna reguła nie odtworzyła całego oryginału, choć przy 20% szumu końcowe obrazy miały około 90% zgodnych pikseli.

### 4.4. Zbiory z grupami podobnych wzorców

**Metoda.** Z `large-25x25.plus` wybraliśmy podzbiory po 6 i 8 obrazów. Podzbiory zróżnicowane budowaliśmy zachłannie, ograniczając podobieństwo nowego wzorca do już wybranych. Zbiory z grupami budowaliśmy z rozłącznych par najbardziej podobnych obrazów.

| Liczba | Wariant | Średnie podobieństwo par | Hebb stabilne | Dokładny odzysk Hebba | Dokładny odzysk Oji |
| ---: | --- | ---: | ---: | ---: | ---: |
| 6 | zróżnicowany | 0,074 | 6/6 | 100% | 100% |
| 6 | grupy | 0,358 | 0/6 | 0% | 100% |
| 8 | zróżnicowany | 0,093 | 8/8 | 90% | 100% |
| 8 | grupy | 0,292 | 0/8 | 0% | 100% |

Odzysk zmierzyliśmy przy 20% szumu w trybie asynchronicznym. Reguła Oji zachowała wszystkie wzorce w czterech podzbiorach.

### 4.5. Losowe wejście

**Metoda.** Dla każdej konfiguracji podaliśmy 100 niezależnych wejść. Każdy piksel losowaliśmy jako +1 lub −1 z równym prawdopodobieństwem.

| Zbiór i konfiguracja | Oryginał | Negatyw | Stan pasożytniczy | Cykl |
| --- | ---: | ---: | ---: | ---: |
| Zwierzęta, Hebb async | 2 | 1 | 97 | 0 |
| Zwierzęta, Oja async | 42 | 37 | 21 | 0 |
| Litery 14×20, Oja sync | 0 | 0 | 1 | 99 |
| Litery 14×20, Oja async | 0 | 0 | 100 | 0 |

**Interpretacja.** Losowe wejście może prowadzić do oryginału, jego negatywu, stanu pasożytniczego lub cyklu. W tabeli stan pasożytniczy oznacza punkt stały spoza zbioru wzorców uczących i ich negatywów. Przy zerowym pobudzeniu zewnętrznym odwrócenie wszystkich znaków odwraca pola i nie zmienia energii, więc negatyw punktu stałego też jest punktem stałym [6, slajd 25].

### 4.6. Własne wzorce 5×5 i cykl dwustanowy

**Zbiór sprawdzony programem.** Plik `src/own/stable-5x5.csv` zawiera 19 wzorców. Po uczeniu regułą Hebba stabilnych było 19 z 19, a po uczeniu regułą Oji 10 z 19. Minimalny margines Hebba wyniósł $2/25$ po uwzględnieniu normalizacji wag.

**Konstrukcja ściśle stabilnego zbioru.** W pierwszych 22 pozycjach wybieramy dowolne 11 pikseli o wartości $+1$. Pozostałe mają wartość $-1$. Ostatnie trzy piksele mają wspólny znak, wybierany niezależnie od pierwszego bloku.

$$
\begin{aligned}
N &= 2\binom{22}{11}=1\,410\,864,\\
x_i^{(q)}z_i &=
\begin{cases}
\dfrac{N}{21\cdot 25}, & 1\le i\le 22,\\[4pt]
\dfrac{2N}{25}, & 23\le i\le 25.
\end{cases}
\end{aligned}
$$

W każdym wzorcu pierwsze 22 piksele sumują się do zera. Stąd średni iloczyn dwóch różnych pozycji z tego bloku wynosi $-1/21$. W ostatniej trójce wynosi $1$, a między blokami $0$. Oba marginesy są dodatnie. To dowód dla całej rodziny, dający dolne ograniczenie na maksymalną liczność ściśle stabilnego zbioru.

Jeśli dopuszczamy zerową preaktywację, wzorce z dokładnie 12 lub 13 pikselami $+1$ dają $2\binom{25}{12}=10\,400\,600$ punktów stałych. Ujemna waga między dowolnymi dwoma pikselami daje wtedy zerowe pole na pikselach większościowych i pole o właściwym znaku na pozostałych. Formalne maksimum to wszystkie $2^{25}=33\,554\,432$ bitmapy. Wówczas wkłady się znoszą, $\mathbf W=\mathbf 0$, więc każde uszkodzenie także pozostaje bez zmian.

**Cykl.** Sieć uczymy regułą Hebba na krzyżu i znaku X. Na wejście podajemy ramkę. Znak `#` oznacza czarny piksel.

```text
Wejście    Krok 1    Krok 2
#####      .#.#.     #####
#...#      #####     #...#
#...#      .#.#.     #...#
#...#      #####     #...#
#####      .#.#.     #####
```

W obu stanach iloczyn skalarny z każdym wzorcem uczącym wynosi $-9$, więc $m_1=m_2=-9/25$. Na 16 pozycjach, gdzie krzyż i X mają przeciwne znaki, ich wkłady się znoszą. Zostaje $z_i=-2a_i/25$, co odwraca znak piksela. Pozostałe dziewięć pikseli zachowuje stan. Tryb asynchroniczny na tym samym wejściu kończy się w negatywie znaku X.

### 4.7. Bitmapy 200×300

**Dane.** Użyliśmy fotografii `camera`, `coins` i `moon` z pakietu scikit-image 0.24.0 [7]. Po konwersji do skali szarości wykadrowaliśmy obrazy centralnie i przeskalowaliśmy je filtrem Lanczosa do 200×300. Przy binaryzacji użyliśmy progów jasności odpowiednio 152, 89 i 112. Średnie podobieństwo par wyniosło 0,159. Po uczeniu każdą z reguł wszystkie trzy wzorce były stabilne.

| Reguła i tryb | Dokładne przy 0–40% szumu | Dokładne przy 50% | Zgodne piksele przy 50% |
| --- | ---: | ---: | ---: |
| Hebb sync | 30/30 na poziom | 0/30 | 52% |
| Hebb async | 30/30 na poziom | 0/30 | 61% |
| Oja sync | 30/30 na poziom | 0/30 | 53% |
| Oja async | 30/30 na poziom | 0/30 | 61% |

Po odwróceniu $d$ z $n$ pikseli wzorca $q$ jego podobieństwo do wejścia $\tilde{\mathbf x}$ wynosi [6, slajd 23]

$$
m_q=\frac{\bigl(\mathbf x^{(q)}\bigr)^\top\tilde{\mathbf x}}{n}=1-\frac{2d}{n}.
$$

Przy 40% szumu $m_q=0{,}2$, a przy 50% spada do zera.

## 5. Wnioski

- W badanych zbiorach **Oja częściej zachowywała wzorce jako punkty stałe**. Komplet 26 liter pozostawał niestabilny dla obu reguł.
- Próba z losowym blokiem 25×50 pokazała, że **wspólne tło może obniżyć odzysk**, mimo większej liczby neuronów.
- **Stabilność i dokładny odzysk mierzą różne własności.** Wynik może mieć większość poprawnych pikseli, a mimo to przypominać inny wzorzec lub stan pasożytniczy.
- Tryb aktualizacji wpływał na końcowy stan. Ta sama ramka 5×5 dawała cykl w trybie synchronicznym i punkt stały w asynchronicznym.

## 6. Uruchomienie

W głównym katalogu repozytorium wykonujemy:

```sh
cargo test --locked
cargo run --release -- 1 2 3 4 5 6 7
```

Aby obejrzeć cykl dwustanowy, generujemy HTML i otwieramy go w przeglądarce.

```sh
cargo run --release -- viz src/own/oscillation-train-5x5.csv 0 hebb sync 0 target/cycle.html src/own/oscillation-input-5x5.csv
```


[Instrukcja programu](docs/launch.md) · [Pełny zapis wyników](docs/results.txt)

## Literatura

1. J. J. Hopfield (1982), *Neural networks and physical systems with emergent collective computational abilities*, PNAS 79(8), 2554–2558. [Artykuł](https://pmc.ncbi.nlm.nih.gov/articles/PMC346238/)
2. D. J. Amit, H. Gutfreund, H. Sompolinsky (1985), *Storing Infinite Numbers of Patterns in a Spin-Glass Model of Neural Networks*, Physical Review Letters 55(14), 1530–1533. [Artykuł](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.55.1530)
3. E. Oja (1982), *Simplified neuron model as a principal component analyzer*, Journal of Mathematical Biology 15, 267–273. [Artykuł](https://link.springer.com/article/10.1007/BF00275687)
4. E. Oja (1989), *Neural Networks, Principal Components, and Subspaces*, International Journal of Neural Systems 1(1), 61–68. [DOI](https://doi.org/10.1142/S0129065789000475)
5. E. Goles, J. Olivos (1980), *Periodic behaviour of generalized threshold functions*, Discrete Mathematics 30(2), 187–189. [DOI](https://doi.org/10.1016/0012-365X(80)90121-1)
6. M. Małkiński, *Neural Networks — Lecture 5: Hopfield Model and Associative Memories*, Politechnika Warszawska. Slajdy 3, 6, 8–9, 13, 16, 18–19, 22–23, 27–28 i 33.
7. Zespół scikit-image, *skimage.data* (dokumentacja wersji 0.24.x). Opisy obrazów `camera`, `coins` i `moon`. [Dokumentacja scikit-image](https://scikit-image.org/docs/0.24.x/api/skimage.data.html)
