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
| Użyty symbol | Znaczenie |
| --- | --- |
| $n$, $i,j,k$ | Liczba neuronów oraz ich indeksy - jeden piksel odpowiada jednemu neuronowi |
| $\mathbf a^{(t)}$ | Stan sieci w chwili $t$ - każdy element $a_i^{(t)}$ przyjmuje wartości $+1$ albo $-1$ (w zależności od koloru) |
| $\tilde{\mathbf x}$ | Wzorzec wejściowy |
| $d$ | Liczba odwróconych pikseli w wejściu |
| $\mathbf x^{(p)}$, $N$, $\mathbf X$ | Wzorzec uczący $p$, liczba wzorców i macierz $\mathbf X=[\mathbf x^{(1)},\ldots,\mathbf x^{(N)}]$, w której każdy wzorzec to jedna kolumna |
| $\mathbf W$ | Macierz wag |
| $z_i=\sum_j w_{ij}a_j$ | Preaktywacja neuronu $i$ - znak sumy decyduje o nowym stanie neuronu |
| $m_p=\frac1n(\mathbf x^{(p)})^\top\mathbf a$ | Podobieństwo stanu do wzorca $p$, gdzie $-1$ to negatyw, $0$ to brak związku (zgadza się połowa pikseli), a $1$ to identyczność |
| $o_{pq}=\frac1n(\mathbf x^{(p)})^\top\mathbf x^{(q)}$ | Podobieństwo dwóch wzorców - w tabelach podawane jako średnia $\lvert o_{pq}\rvert$ po wszystkich parach |
| $\alpha=N/n$ | Obciążenie pamięci |
| $\mathbf y=\mathbf W\mathbf x$, $\mathbf v=\mathbf W^\top\mathbf y$, $\eta$ | Odpowiedź sieci na wzorzec, rekonstrukcja wzorca i krok uczenia w regule Oji |
| $E(\mathbf a)=-\frac12\mathbf a^\top\mathbf W\mathbf a$ | Energia stanu $\mathbf a$ |

| Pojęcie | Znaczenie |
| --- | --- |
| Punkt stały (stan stabilny) | Sytuacja, w której stan żadnego neuronu się nie zmienia - wzorzec jest zapamiętany, jeśli jest punktem stałym |
| Negatyw | Wzorzec z odwróconymi kolorami |
| Stan pasożytniczy | Punkt stały, który nie jest żadnym wzorcem ani negatywem - zwykle mieszanina kilku wzorców |
| Cykl | W trybie synchronicznym sieć przechodzi na zmianę między dwoma stanami i nie zatrzymuje się |

## Cel projektu
Celem projektu jest sprawdzenie działania zaimplementowanej sieci Hopfielda. Sieć ta działa jak pamięć skojarzeniowa: zapamiętane bitmapy są jej punktami stałymi, a uszkodzony obraz podany na wejście zmienia się krok po kroku, aż sieć zatrzyma się w punkcie stałym. Porównano dwie reguły uczenia (Hebba i Oji) oraz dwa tryby aktualizacji neuronów (synchroniczny i asynchroniczny).

## Główne wyniki
Na ośmiu dostarczonych zbiorach Hebb zachował jako punkty stałe 12 z 74 wzorców, a Oja 48 z 74 – wszystkie wzorce, których Oja nie zapamiętała, pochodzą z jednego zbioru 26 liter. Obciążenie $\alpha$ we wszystkich zbiorach wynosiło od 0,005 do 0,10, czyli poniżej teoretycznej pojemności 0,138, więc o porażkach Hebba decydowało podobieństwo wzorców, a nie ich liczba. Dzięki obliczeniom bez pełnej macierzy wag sieć działała także na dużych bitmapach np. 200×300 (60 000 neuronów).

## 1. Aktualizacja neuronów
Neuron $i$ sumuje stany pozostałych neuronów ważone wagami połączeń. Dodatnia waga $w_{ij}$ oznacza ten sam kolor neuronów $i$ oraz $j$ ujemna – kolor przeciwny. Nowy stan to znak tej sumy, a przy wyniku 0.0 neuron zachowuje dotychczasowy stan.

$$
z_i^{(t+1)} = \sum_{j=1}^{n} w_{ij} a_j^{(t)},
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
| Synchroniczny | Wszystkie neurony liczą $z_i$ ze stanu z początku kroku i zmieniają się jednocześnie | Stan się nie zmienił (punkt stały) albo jest równy stanowi sprzed dwóch kroków (cykl) |
| Asynchroniczny | Neurony są aktualizowane po kolei, w stałej kolejności pikseli (wierszami) oraz każdy neuron widzi już zmiany wcześniejszych | Pełne przejście bez żadnej zmiany |

### 1.1 Zbieżność trybu asynchronicznego
Przy symetrycznych wagach i zerowej przekątnej zmiana stanu jednego neuronu $k$ zmienia energię o

$$
E(\mathbf a) = -\frac12 \mathbf a^\top\mathbf W\mathbf a,
\qquad
\Delta E = -\Delta a_k\,z_k,
\qquad
\Delta a_k = a_k^{(t+1)}-a_k^{(t)}.
$$

Neuron zmienia się tylko na znak $z_k$, więc $\Delta a_k$ i $z_k$ mają ten sam znak i $\Delta E<0$. Energia maleje przy każdej zmianie, a stanów jest skończenie wiele, więc sieć zatrzymuje się w punkcie stałym i nie może wrócić do stanu, w którym już była [1, 6, slajdy 27–28].

### 1.2 Cykle w trybie synchronicznym
Każdy neuron decyduje na podstawie starego stanu. Argument o malejącej energi przestaje więc działać. Przy symetrycznych wagach sieć synchroniczna kończy w punkcie stałym albo w cyklu długości 2 [5, 6, slajd 19], dlatego do wykrycia cyklu wystarcza porównanie ze stanem sprzed dwóch kroków.

$$
E(\mathbf a) = -\frac12 \mathbf a^\top\mathbf W\mathbf a - \mathbf b^\top\mathbf a,
\qquad
\Delta E = -\Delta a_k\,z_k,
\qquad
\Delta a_k = a_k^{(t+1)}-a_k^{(t)}.
$$

## 2. Reguły uczenia

### 2.1 Reguła Hebba
**Wzór:** Waga $w_{ij}$ rośnie, gdy piksele $i$ i $j$ mają we wzorcach ten sam kolor, i maleje, gdy mają różne [6, slajd 13].

$$
w_{ij} =
\begin{cases}
\displaystyle \frac{1}{n}\sum_{p=1}^{N} x_i^{(p)}x_j^{(p)}, & i\ne j,\\
0, & i=j,
\end{cases}
\qquad
\mathbf W=\frac{1}{n}\left(\mathbf X\mathbf X^\top-N\mathbf I\right).
$$

Uczenie to jeden przebieg po wzorcach, bez epok i bez sprawdzania wyniku.

**Wyjaśnienie działania:** Po podstawieniu wag Hebba do preaktywacji każdy wzorzec głosuje na swoją wartość piksela $i$, z siłą równą podobieństwu $m_p$ obecnego stanu do tego wzorca. Wygrywa wzorzec najbardziej podobny do stanu. Ostatni składnik usuwa głos neuronu na samego siebie, co odpowiada zerowej przekątnej [1, 6, slajd 16].

$$
z_i=\sum_{p=1}^{N}m_p\, x_i^{(p)}-\frac{N}{n}\,a_i.
$$

**Sygnał i przesłuch:** Gdy na wejściu jest wzorzec uczący, $\mathbf a=\mathbf x^{(q)}$, mamy $m_q=1$ i $m_p=o_{pq}$ dla pozostałych wzorców:

$$
z_i = \underbrace{\Bigl(1-\frac Nn\Bigr)x_i^{(q)}}_{\text{sygnał}}
      + \underbrace{\sum_{p\ne q} o_{pq}\, x_i^{(p)}}_{\text{przesłuch}}.
$$

Piksel pozostaje bez zmian, gdy $x_i^{(q)}z_i\ge 0$ (przy $0$ decyduje reguła zachowania stanu). Najbardziej zagrożone są piksele, którymi wzorzec $q$ różni się od wzorców do niego podobnych: tam każdy taki wzorzec odejmuje od sygnału swoje $o_{pq}$. Jeśli suma tych podobieństw przekroczy $1-N/n$, piksel się odwraca [6, slajdy 18 i 22].

Dla wzorców losowych $o_{pq}$ jest rzędu $1/\sqrt n$, przesłuch zachowuje się jak szum, a pamięć działa do obciążenia $\alpha_c\approx 0{,}138$ (z niewielkim odsetkiem błędnych pikseli). W dostarczonych zbiorach $o_{pq}$ wynosi od kilkunastu do kilkudziesięciu setnych, więc do zniszczenia wzorca wystarczy jeden lub dwa podobne wzorce, niezależnie od $\alpha$.

### 2.2 Reguła Oji
Dla jednego neuronu z wektorem wag $\mathbf w$ i odpowiedzią $y=\mathbf w^\top\mathbf x$ reguła Oji dodaje do składnika Hebba człon hamujący, proporcjonalny do kwadratu odpowiedzi. Bez niego wagi rosłyby przy każdej epoce. Z nim wektor wag zbiega do głównej składowej danych [3].

$$
\Delta\mathbf w = \eta y\mathbf x - \eta y^2\mathbf w.
$$

**Wariant użyty w projekcie:** Dla sieci z wieloma neuronami stosujemy wersję podprzestrzenną [4]. Dla każdego wzorca uczącego $\mathbf x=\mathbf x^{(p)}$:

$$
\begin{aligned}
\mathbf y &= \mathbf W\mathbf x
   && \text{odpowiedź sieci na wzorzec},\\
\mathbf v &= \mathbf W^\top\mathbf y
   && \text{rekonstrukcja wzorca},\\
w_{ij} &\leftarrow w_{ij}+\eta\,y_i(x_j-v_j)
   && \text{poprawka dla } i\ne j.
\end{aligned}
$$

Uczenie startuje od macierzy Hebba podzielonej przez największą sumę modułów w wierszu. Po tej normalizacji $\mathbf W\mathbf x$ nie jest na starcie większe od $\mathbf x$, więc pierwsze poprawki nie rosną lawinowo. Krok $\eta=0{,}5/n$ jest dzielony przez $n$, bo jedna poprawka zmienia odpowiedź sieci około $n$ razy silniej (suma po $n$ pikselach). Po każdej epoce macierz jest symetryzowana, $\mathbf W\leftarrow(\mathbf W+\mathbf W^\top)/2$, bo poprawki $y_i(x_j-v_j)$ nie są symetryczne, a bez symetrii tryb asynchroniczny traci gwarancję zbieżności. Uczenie kończy się po zadanym parametrem epokach lub gdy największa poprawka spadnie poniżej $10^{-6}$.

**Wyjaśnienie działania:** Poprawka jest proporcjonalna do błędu rekonstrukcji $\mathbf x-\mathbf v$, podobnie jak w regule delta, tylko celem jest sam wzorzec. Wagi przestają się zmieniać dopiero wtedy, gdy sieć odtwarza każdy wzorzec uczący. Regule Hebba brakuje tego sprzężenia zwrotnego: dodaje korelacje niezależnie od tego, czy wzorzec jest już stabilny. Idealnym celem uczenia jest macierz rzutu na podprzestrzeń rozpiętą przez wzorce (reguła rzutowa) [6, slajd 37]. Dla niej $\mathbf W\mathbf x^{(p)}=\mathbf x^{(p)}$, więc przesłuch jest skompensowany niezależnie od podobieństwa wzorców.

**Ograniczenia:** W projekcie wymuszamy zerową przekątną i symetrię wag. Przy wielu podobnych wzorcach albo dużym obciążeniu macierz spełniająca te warunki i odtwarzająca wszystkie wzorce może nie istnieć. Uczenie zbiega wtedy do kompromisu i część wzorców pozostaje niestabilna (zbiór 26 liter w sekcji 4.1, zbiór 5×5 w sekcji 4.6).

### 2.3 Obliczenia dla dużych bitmap
Pełna macierz wag dla 60 000 neuronów zajmowałaby zapisana w standardowym 32-bitowym float'cie około 14,4 GB. Dla obrazów powyżej 2048 pikseli program używa więc jednej z dwóch reprezentacji, które dają te same wyniki co pełna macierz.

- **Hebb:** przechowuje tylko wzorce ($N\cdot n$ liczb zamiast $n^2$) i liczy pole lokalne bezpośrednio z głosowania wzorców. Wspólny dodatni czynnik $1/n$ nie zmienia znaku, więc jest pomijany. W trybie asynchronicznym, po zmianie neuronu $k$, program nie przelicza iloczynów skalarnych od nowa, tylko zwiększa każde $u_p$ o $x_k^{(p)}\Delta a_k$.

$$
u_p=\bigl(\mathbf x^{(p)}\bigr)^\top\mathbf a = n\,m_p,
\qquad
z_i=\frac{1}{n}\left(\sum_{p=1}^{N}x_i^{(p)}u_p-N a_i\right).
$$

- **Oja:** korzysta z faktu, iż piksele o tych samych wartościach we wszystkich wzorcach są dla reguły uczenia nierozróżnialne. Dostają identyczne wagi po regule Hebba i identyczne poprawki w każdej epoce Oji. Program trzyma więc jeden wiersz wag na grupę takich pikseli (piksele o przeciwnych wartościach trafiają do tej samej grupy z odwróconym znakiem), a każdy piksel zachowuje własny stan. Grup jest najwyżej $2^{N-1}$, niezależnie od $n$: przy trzech wzorcach to cztery grupy. Pole o module poniżej $10^{-10}$ traktowane jest jako remis.

## 3. Mierzona skuteczność

| Metryka | Sposób liczenia |
| --- | --- |
| Stabilność wzorca | Czysty wzorzec uczący podany na wejście nie zmienia się w pierwszym kroku synchronicznym, czyli jest punktem stałym |
| Dokładny odzysk | Sieć kończy w punkcie stałym identycznym z oryginałem co do piksela - cykl i każdy inny stan końcowy zaliczona jako porażka |
| Zgodność pikseli | Odsetek pikseli równych oryginałowi w stanie końcowym, liczony także po nieudanym odzysku - wysoka zgodność nie musi oznaczać dobrego odtworzenia, jeśli wzorce mają dużo wspólnych pikseli - stan pasożytniczy podobny do wszystkich wzorców potrafi mieć dużą zgodność |

**Uszkodzenia.** Szum $k\%$ polega na odwróceniu dokładnie $\mathrm{round}(k\cdot n)$ losowo wybranych pikseli, dla $k$ równego 0, 10, 20, 30, 40 i 50. Przy 50% wejście jest tak samo podobne do wzorca jak do jego negatywu, więc odtworzenie nie jest możliwe. Drugi rodzaj uszkodzenia to biały poziomy pas na całą szerokość obrazu, o wysokości 20% wierszy, w losowym miejscu.

**Próby.** Każdy wzorzec zaszumiony został 10 razy dla każdego poziomu, więc dla zbioru z $N$ wzorcami jeden wynik to $10N$ prób. Generator zaczynał od ziarna 42. W sekcjach 4.2–4.5 kolejne próby korzystały z kolejnych realizacji generatora, a w sekcji 4.7 każda reguła i każdy tryb dostały te same uszkodzone wejścia. [Pełne wyniki przeprowadzonych testów](docs/results.txt)

**Opis zbiorów.** Podobieństwo zbioru to średnia $\lvert o_{pq}\rvert$ po wszystkich parach różnych wzorców. Aktywność to udział czarnych pikseli we wszystkich wzorcach zbioru. Wspólne piksele to odsetek pozycji, które mają ten sam kolor we wszystkich wzorcach.

## 4. Eksperymenty

### 4.1. Stabilność dostarczonych wzorców
**Metoda:** Dla każdego zbioru uczyliśmy obie sieci na komplecie wzorców i podawaliśmy na wejście każdy oryginał bez szumu.

| Zbiór | $n$ | $N$ | $\alpha$ | Aktywność | Wspólne piksele | Podobieństwo | Hebb | Oja |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OCRA-12x30-cut | 360 | 7 | 0,019 | 0,360 | 28,1% | 0,272 | 1/7 | 7/7 |
| animals-14x9 | 126 | 6 | 0,048 | 0,311 | 16,7% | 0,212 | 2/6 | 6/6 |
| large-25x25 | 625 | 6 | 0,010 | 0,316 | 10,9% | 0,142 | 5/6 | 6/6 |
| large-25x25.plus | 625 | 15 | 0,024 | 0,300 | 1,1% | 0,171 | 1/15 | 15/15 |
| large-25x50 | 1250 | 6 | 0,005 | 0,658 | 55,4% | 0,567 | 0/6 | 6/6 |
| letters-14x20 | 280 | 26 | 0,093 | 0,144 | 46,4% | 0,728 | 0/26 | 0/26 |
| letters-abc-8x12 | 96 | 3 | 0,031 | 0,122 | 85,4% | 0,806 | 0/3 | 3/3 |
| small-7x7 | 49 | 5 | 0,102 | 0,416 | 4,1% | 0,286 | 3/5 | 5/5 |

**Wyniki dla uczenia metodą Hebba:** 

Obciążenie wszystkich zbiorów wynosi od 0,005 do 0,102, czyli mniej niż 0,138, ale nie został zapamiętany komplet w żadnym zbiorze. Najlepiej wypadł zbiór o najmniejszym podobieństwie (`large-25x25`, 5/6), a żadnego wzorca sieć nie zachowała w zbiorach najbardziej podobnych (litery, `large-25x50`). Zgadza się to ze wzorem na przesłuch z sekcji 2: pojemność 0,138 dotyczy wzorców losowych [2], dla których $o_{pq}$ jest rzędu $1/\sqrt n$, a tutaj podobieństwa sięgają kilkudziesięciu setnych. Wysokie podobieństwo bierze się ze wspólnego tła: w zbiorze ABC 85,4% pikseli ma ten sam kolor we wszystkich wzorcach, a w literach czarnych pikseli jest tylko 14,4%.

W zbiorze liter każdy wzorzec ma inną literę o podobieństwie od 0,66 do 0,99 i każdy jest niestabilny. Bliski sąsiad nie wyjaśnia jednak wszystkiego: w `large-25x25` najbardziej podobną parą są wzorce 1 i 3 ($o=0{,}47$), a niestabilny jest tylko wzorzec 3. Przesłuch sumuje się ze wszystkich pozostałych wzorców, więc o stabilności pojedynczego wzorca decyduje cały zbiór.

Przy uczeniu na pierwszych $k$ wzorcach zbioru dodanie wzorca potrafi zniszczyć wzorce wcześniej stabilne. W `OCRA` przy $k=4$ stabilne są 4 wzorce, a przy $k=5$ już tylko 2. W `large-25x25.plus` przy $k=6$ stabilnych jest 5, a przy $k=7$ tylko 2.

**Wyniki dla uczenia metodą Oji:**

Reguła Oji zachowała komplet w siedmiu zbiorach, także w zbiorze ABC o najwyższym podobieństwie. Wyjątkiem są litery 14×20: przy uczeniu na pierwszych $k$ literach Oja utrzymuje komplet do $k=4$, a od $k=16$ żadna litera nie jest stabilna. Dla Oji liczy się więc zarówno podobieństwo, jak i liczba wzorców, co odpowiada ograniczeniu opisanemu w sekcji 2.

### 4.2. Ten sam wzorzec w rozmiarze 25×25 i 25×50
**Metoda:** Górne połówki wzorców 25×50 są identyczne z odpowiednimi obrazami 25×25, a dolne połówki są w całości czarne. Wspólny blok dodaje do iloczynu skalarnego każdej pary 625 jedynek, więc

$$
o_{pq}^{(25\times 50)} = \frac{o_{pq}^{(25\times 25)}+1}{2}.
$$

Zmierzone podobieństwa zgadzają się z tym wzorem dla wszystkich 15 par z dokładnością do 0,001. Na przykład wzorce 0 i 2 w wersji 25×25 są praktycznie niezwiązane ($o=-0{,}011$), a w wersji 25×50 mają podobieństwo 0,494. W wariancie kontrolnym dolną połowę zastąpiliśmy pikselami losowanymi niezależnie dla każdego wzorca. Losowe połówki zgadzają się średnio w połowie pikseli, więc nie dodają podobieństwa, a dzielą je mniej więcej przez dwa.

| Wariant | $n$ | Podobieństwo | Hebb stabilne | Hebb odzysk sync / async | Oja stabilne | Oja odzysk sync / async |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 25×25 | 625 | 0,142 | 5/6 | 67% / 50% | 6/6 | 100% / 100% |
| 25×50, wspólny blok | 1250 | 0,567 | 0/6 | 0% / 0% | 6/6 | 100% / 100% |
| 25×50, losowy blok | 1250 | 0,076 | 6/6 | 100% / 100% | 6/6 | 100% / 100% |

Odzysk zmierzono przy 20% szumu. Oba warianty 25×50 mają tę samą liczbę neuronów i wzorców, a różnią się tylko zawartością dolnej połowy. Z losowym blokiem Hebb działa lepiej niż na oryginale 25×25, ze wspólnym nie zachowuje żadnego wzorca. Przyczyną różnicy jest więc wspólny blok, który podnosi podobieństwo każdej pary do co najmniej około 0,5, a nie rozmiar obrazu. Oja jest na to odporna, bo w trakcie uczenia kompensuje przesłuch.

### 4.3. Odzysk w zależności od rodzaju zbioru, liczności i rodzaju uszkodzenia
**Rodzaj zbioru:** Dokładny odzysk w trybie asynchronicznym oraz zgodność pikseli dla Hebba przy 20% szumu.

| Zbiór | $n$ | $N$ | Podobieństwo | Hebb 20% | Zgodność pikseli Hebb 20% | Oja 20% | Oja 30% | Oja 40% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OCRA-12x30-cut | 360 | 7 | 0,272 | 6% | 79% | 100% | 100% | 84% |
| animals-14x9 | 126 | 6 | 0,212 | 13% | 80% | 100% | 100% | 57% |
| large-25x25 | 625 | 6 | 0,142 | 52% | 95% | 100% | 100% | 100% |
| large-25x25.plus | 625 | 15 | 0,171 | 7% | 73% | 100% | 100% | 97% |
| large-25x50 | 1250 | 6 | 0,567 | 0% | 87% | 100% | 100% | 100% |
| letters-14x20 | 280 | 26 | 0,728 | 0% | 91% | 0% | 0% | 0% |
| letters-abc-8x12 | 96 | 3 | 0,806 | 0% | 95% | 90% | 77% | 53% |
| small-7x7 | 49 | 5 | 0,286 | 40% | 91% | 100% | 76% | 18% |

**Wyniki dla uczenia metodą Hebba:** 

Zgodność pikseli prawie nie zależy od poziomu szumu: dla `large-25x50` wynosi 87% przy każdym poziomie od 0 do 40%, a dla liter 91%. Sieć kończy w tym samym stanie pasożytniczym niezależnie od wejścia. Wysoka zgodność wynika ze wspólnego tła tego stanu i wszystkich wzorców, a nie z odtworzenia.

**Wyniki dla uczenia metodą Oji:**

Metoda odtwarza wszystkie zbiory poza literami 14×20 do 20% szumu. Przy 40% rozstrzyga rozmiar obrazu: zbiory z 625 i 1250 pikselami nadal mają 97–100%, a `small-7x7` z 49 pikselami spada do 18%. Przy małym $n$ ta sama część szumu silniej obniża podobieństwo wejścia do oryginału. Zbiór ABC traci skuteczność już przy 20–30% mimo tylko trzech wzorców, bo wzorce są do siebie bardzo podobne.

**Liczność:** Sieć uczona na pierwszych $k$ wzorcach zbioru i zaszumialiśmy każdy z nich w 20% (tryb asynchroniczny).

| $k$ | 1–4 | 5 | 6 | 7 | 10 | 15 | 16–26 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `large-25x25.plus`, Hebb | 100% | 94% | 53% | 29% | 23% | 7% | – |
| `large-25x25.plus`, Oja | 100% | 100% | 100% | 100% | 100% | 100% | – |
| litery 14×20, Hebb | 100% do $k=2$, potem 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| litery 14×20, Oja | 100% | 58% | 50% | 71% | 10% | 6% | 0% |

Metoda Hebba na `large-25x25.plus` zaczyna tracić wzorce przy $k=5$, czyli przy $\alpha=0{,}008$. Na literach wystarczają trzy wzorce. Metoda Oja zachowuje komplet na `large-25x25.plus` dla każdego $k$, a na literach jej skuteczność spada nierówno: zależy od tego, czy dodana litera jest podobna do już zapamiętanych.

**Rodzaj uszkodzenia.** Losowy szum 20% i biały poziomy pas o wysokości 20% wierszy, tryb asynchroniczny.

| Zbiór | Hebb szum | Hebb pas | Oja szum | Oja pas |
| --- | ---: | ---: | ---: | ---: |
| OCRA-12x30-cut | 6% | 4% | 100% | 100% |
| animals-14x9 | 12% | 7% | 100% | 100% |
| large-25x25 | 43% | 33% | 100% | 100% |
| large-25x25.plus | 7% | 0% | 100% | 100% |
| large-25x50 | 0% | 0% | 100% | 100% |
| letters-14x20 | 0% | 0% | 0% | 0% |
| letters-abc-8x12 | 0% | 0% | 93% | 100% |
| small-7x7 | 36% | 44% | 98% | 100% |

Metoda Oja radzi sobie z oboma uszkodzeniami tak samo. Przy metodzie Hebba pas daje zwykle trochę gorszy wynik niż szum, bo usuwa całą informację z jednego fragmentu obrazu. Wyjątki wynikają z tego, że pas zmienia mniej pikseli niż szum: w `small-7x7` 20% z 7 wierszy to jeden wiersz (14% obrazu), a w jasnym zbiorze ABC pas zmienia tylko nieliczne czarne piksele. Każdy wynik to 30–260 prób, więc różnice rzędu 10 punktów procentowych mieszczą się w losowości; z tego powodu wyniki dla tego samego zbioru różnią się też nieznacznie między sekcjami 4.2 i 4.3.

### 4.4. Zbiory z grupami podobnych wzorców
**Metoda:** Z `large-25x25.plus` wybraliśmy podzbiory po 6 i 8 obrazów. Podzbiór zróżnicowany budowano zachłannie, dodając zawsze wzorzec najmniej podobny do już wybranych. Podzbiór z grupami składa się z rozłącznych par najbardziej podobnych obrazów. Oba warianty mają ten sam rozmiar obrazów i pochodzą z tego samego pliku.

| $N$ | Wariant | Podobieństwo | Maks. podobieństwo pary | Hebb stabilne | Odzysk Hebb | Odzysk Oja |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 6 | zróżnicowany | 0,074 | 0,267 | 6/6 | 100% | 100% |
| 6 | z grupami | 0,358 | 0,498 | 0/6 | 0% | 100% |
| 8 | zróżnicowany | 0,093 | 0,306 | 8/8 | 90% | 100% |
| 8 | z grupami | 0,292 | 0,498 | 0/8 | 0% | 100% |

Odzysk mierzono przy 20% szumu w trybie asynchronicznym. Przy tej samej liczności Hebb zachowuje wszystkie wzorce zbioru zróżnicowanego i żadnego ze zbioru z grupami. Średnie podobieństwo zbioru z grupami jest przy ośmiu wzorcach niższe niż przy sześciu, a wynik się nie zmienia: o porażce decydują pary bardzo podobnych wzorców, a nie średnia.

We wszystkich 60 i 80 próbach Hebba na zbiorach z grupami sieć kończyła w stanie pasożytniczym. Ani razu nie trafiła w drugi wzorzec z pary. Podobne wzorce nie są więc mylone ze sobą, tylko zlewają się we wspólny stan mieszany, który nie jest żadnym z nich. Oja odtworzyła oryginał we wszystkich próbach.

### 4.5. Losowe wejście
**Metoda:** Dla każdej konfiguracji podaliśmy 100 niezależnych wejść, w których każdy piksel ma wartość $+1$ lub $-1$ z równym prawdopodobieństwem. Tabela dotyczy trybu asynchronicznego.

| Zbiór | Hebb: wzorzec lub negatyw | Hebb: pasożytniczy | Hebb: różnych stanów | Oja: wzorzec lub negatyw | Oja: pasożytniczy | Oja: różnych stanów |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OCRA-12x30-cut | 2 | 98 | 10 | 49 | 51 | 59 |
| animals-14x9 | 3 | 97 | 4 | 79 | 21 | 31 |
| large-25x25 | 12 | 88 | 21 | 76 | 24 | 36 |
| large-25x25.plus | 0 | 100 | 2 | 81 | 19 | 46 |
| large-25x50 | 0 | 100 | 10 | 63 | 37 | 45 |
| letters-14x20 | 0 | 100 | 2 | 0 | 100 | 71 |
| letters-abc-8x12 | 0 | 100 | 2 | 88 | 12 | 8 |
| small-7x7 | 66 | 34 | 14 | 45 | 55 | 45 |


**Wyniki dla uczenia metodą Hebba:**

W czterech zbiorach wszystkie 100 wejść skończyło w jednym z dwóch stanów: stanie pasożytniczym i jego negatywie. Stany pasożytnicze są bardzo podobne do wzorców (średnie podobieństwo do najbliższego wzorca od 0,68 do 0,94), więc są mieszaninami wzorców, które zajęły miejsce samych wzorców. Wzorce niestabilne z sekcji 4.1 nie przyciągają żadnego wejścia: w `small-7x7` wzorzec 2 przyciągnął 60 wejść, a niestabilne wzorce 3 i 4 żadnego.

**Wyniki dla uczenia metodą Oji:**

Poza literami 14×20 od 45 do 88 wejść kończyło we wzorcu lub negatywie, a liczba trafień rozkładała się na wszystkie wzorce (w `large-25x25` od 8 do 19 na wzorzec). Trafienia we wzorce i w negatywy były zbliżone liczebnie, na przykład 44 i 32 w `large-25x25`. Odwrócenie wszystkich stanów odwraca znaki wszystkich pól, więc negatyw wzorca jest punktem stałym, jeśli wzorzec nim jest, a losowe wejście trafia do nich z podobną częstością.

**Tryby aktualizacji:** 

Cykle pojawiały się najczęściej dla liter 14×20 po regule Oji (99 na 100 wejść) i dla `small-7x7` po regule Oji (25). Tryb asynchroniczny potrzebował mniej kroków, na przykład 4,4 zamiast 10,4 dla `large-25x25.plus` po regule Oji.

### 4.6. Własne wzorce 5×5 i cykl dwustanowy
**Zbiór stabilny.** Proponujemy zbiór 19 wzorców 5×5 (plik `src/own/stable-5x5.csv`), dobranych tak, by były możliwie niepodobne: każda para zgadza się na 13 z 25 pikseli, czyli $o_{pq}=1/25$, najmniej, ile się da przy nieparzystym $n$. Przesłuch jest wtedy bardzo mały i reguła Hebba utrzymuje wszystkie wzorce przy obciążeniu 0,76.

| Zbiór | Liczba wzorców | $\alpha$ | Hebb stabilne | Oja stabilne |
| --- | ---: | ---: | ---: | ---: |
| Pojemność dla wzorców losowych | ≈ 3,5 | 0,138 | – | – |
| Losowe wektory dokładane zachłannie | 13 | 0,52 | 13/13 | – |
| Proponowany zbiór | 19 | 0,76 | 19/19 | 10/19 |

Przy wzorcach możliwie niepodobnych reguła Hebba mieści kilka razy więcej wzorców, niż wynika z pojemności dla wzorców losowych. Oja przy tak dużym obciążeniu zbiega do kompromisu opisanego w sekcji 2 i traci 9 wzorców.

**Liczniejszy zbiór, stabilny bez pamięci.** Jeszcze więcej stabilnych wzorców daje rodzina $2\binom{22}{11}=1\,410\,864$ bitmap, w których pierwsze 22 piksele są w połowie czarne, a ostatnie trzy mają jednakowy kolor. Po uczeniu Hebbem każda z nich jest punktem stałym, ale wagi zapamiętują tylko tę regułę, a nie konkretne obrazy: po odwróceniu jednego piksela sieć poprawia pierwszy w kolejności piksel, który przywraca regułę, i zwykle kończy w innym wzorcu niż oryginał. Stabilność wzorców nie wystarcza więc do działania pamięci.

**Cykl.** Hebba uczymy na krzyżu i znaku X, a na wejście podajemy ramkę. Znak `#` oznacza czarny piksel.

```text
Wejście    Krok 1    Krok 2
#####      .#.#.     #####
#...#      #####     #...#
#...#      .#.#.     #...#
#...#      #####     #...#
#####      .#.#.     #####
```

Ramka i stan z kroku 1 są tak samo podobne do krzyża, jak do X ($m_{\text{krzyż}}=m_{\text X}=-9/25$). Na 16 pikselach, na których krzyż i X się różnią, ich głosy się znoszą i każdy z tych pikseli dąży do odwrócenia. W trybie synchronicznym wszystkie 16 odwraca się jednocześnie, remis się powtarza i w następnym kroku piksele wracają. W trybie asynchronicznym pierwsze odwrócenie przechyla remis na korzyść jednego wzorca i sieć w jednym przejściu kończy w negatywie znaku X.

### 4.7. Bitmapy 200×300
**Dane:** Użyliśmy fotografii `camera`, `coins` i `moon`. Średnie podobieństwo par wyniosło 0,159. Zarówno reguła Hebba jak i Oja zachowały 3/3 wzorców.

| Reguła i tryb | Dokładny odzysk przy 0–40% szumu | Dokładny odzysk przy 50% | Zgodność pikseli przy 50% |
| --- | ---: | ---: | ---: |
| Hebb sync | 30/30 na poziom | 0/30 | 52% |
| Hebb async | 30/30 na poziom | 0/30 | 61% |
| Oja sync | 30/30 na poziom | 0/30 | 53% |
| Oja async | 30/30 na poziom | 0/30 | 61% |

Po odwróceniu $d$ z $n$ pikseli wzorca $q$ jego podobieństwo do wejścia wynosi [6, slajd 23]

$$
m_q=\frac{\bigl(\mathbf x^{(q)}\bigr)^\top\tilde{\mathbf x}}{n}=1-\frac{2d}{n}.
$$

Przy 40% szumu $m_q=0{,}2$ i oryginał nadal dominuje nad przesłuchem od dwóch pozostałych obrazów. Przy 50% $m_q=0$: wejście nie niesie informacji o oryginale i sieć trafia w jeden z trzech obrazów albo w negatyw, stąd zgodność pikseli bliska 50–60%. Obciążenie wynosiło $\alpha=0{,}00005$, a podobieństwo obrazów było niewielkie, więc zadanie jest łatwe i obie reguły dały ten sam wynik. Eksperyment pokazuje przede wszystkim, że implementacja działa na dużych bitmapach wejściowych.

## 5. Wnioski

- **Metoda Hebba ograniczana jest przez podobieństwo wzorców, a nie ich liczbę:** We wszystkich dostarczonych zbiorach obciążenie było mniejsze od 0,138, a Hebb zachował tylko 12 z 74 wzorców. Najlepiej działał na zbiorze o najmniejszym podobieństwie, a dodanie podobnego wzorca potrafiło zniszczyć wzorce wcześniej stabilne.

- **Metoda Oja kompensuje przesłuch:** Zachowała 48 z 74 wzorców, w tym komplet we wszystkich zbiorach poza 26 literami. Przy zerowej przekątnej i symetrii wag nie zawsze istnieje macierz odtwarzająca wszystkie wzorce, dlatego Oja przegrywa przy wielu bardzo podobnych wzorcach (litery) i przy dużym obciążeniu (zbiór 5×5: 10/19 wobec 19/19 Hebba).

- **Więcej neuronów pomaga wówczas, gdy niosą informację odróżniającą wzorce:** Wspólny czarny blok w zbiorze 25×50 podniósł podobieństwo każdej pary do $(o+1)/2$ i Hebb stracił wszystkie wzorce. Losowy blok tej samej wielkości przywrócił komplet.

- **Podobne wzorce nie są mylone ze sobą, tylko zlewają się w stany mieszane:** Przy tej samej liczności Hebb zachował 6/6 wzorców zbioru zróżnicowanego i 0/6 zbioru z parami podobnych obrazów. We wszystkich próbach na zbiorach z grupami sieć kończyła w stanie pasożytniczym, ani razu w drugim wzorcu z pary.

- **Odporność na szum zależy od rozmiaru obrazu i podobieństwa wzorców:** Oja odtwarzała wszystkie zbiory poza literami 14×20 przy 20% szumu. Przy 40% zbiory z 625 i 1250 pikselami nadal miały 97–100%, a zbiór 7×7 – 18%. Przy 50% wejście nie niesie informacji o oryginale i żadna reguła go nie odtwarza.

- **Zgodność pikseli jest kontrintuicyjnym znacznikiem jakości:** U Hebba wynosiła 87–95% w zbiorach, w których dokładny odzysk był zerowy, bo sieć niezależnie od wejścia kończyła w tym samym stanie pasożytniczym, podobnym do wszystkich wzorców naraz. Losowe wejścia trafiały tam w 100 na 100 przypadków dla czterech zbiorów, a po regule Oji w większości kończyły we wzorcach lub ich negatywach, z podobną częstością.

- **Stabilność nie wystarcza do działania pamięci.** Zbiór 19 prawie ortogonalnych wzorców 5×5 jest stabilny po regule Hebba przy obciążeniu 0,76. Rodzina ponad miliona wzorców 5×5 też jest stabilna, ale sieć nie wraca do oryginału nawet po jednym odwróconym pikselu.

## Literatura

1. J. J. Hopfield (1982), *Neural networks and physical systems with emergent collective computational abilities*, PNAS 79(8), 2554–2558. [Artykuł](https://pmc.ncbi.nlm.nih.gov/articles/PMC346238/)
2. D. J. Amit, H. Gutfreund, H. Sompolinsky (1985), *Storing Infinite Numbers of Patterns in a Spin-Glass Model of Neural Networks*, Physical Review Letters 55(14), 1530–1533. [Artykuł](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.55.1530)
3. E. Oja (1982), *Simplified neuron model as a principal component analyzer*, Journal of Mathematical Biology 15, 267–273. [Artykuł](https://link.springer.com/article/10.1007/BF00275687)
4. E. Oja (1989), *Neural Networks, Principal Components, and Subspaces*, International Journal of Neural Systems 1(1), 61–68. [DOI](https://doi.org/10.1142/S0129065789000475)
5. E. Goles, J. Olivos (1980), *Periodic behaviour of generalized threshold functions*, Discrete Mathematics 30(2), 187–189. [DOI](https://doi.org/10.1016/0012-365X(80)90121-1)
6. M. Małkiński, *Neural Networks — Lecture 5: Hopfield Model and Associative Memories*, Politechnika Warszawska.
