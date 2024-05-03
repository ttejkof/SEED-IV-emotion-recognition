## Data loading
- [ ] Napraviti klasu pomocu koje moze da se iterira po svim videima za jednog coveka
- [ ] Napraviti klasu pomocu koje moze da se iterira po svim videima jednog sessiona
- [ ] Napraviti klasu pomocu koje moze da se iterira po svim videima  - **Drakula** 
- [ ] Konvertovati fajlove u fif format radi jednostavnosti - **Drakula** 
## Preprocesing 
- [ ] Proveriti da li ima artefkata i ako ih ima videti da li moze da ih se resimo
- [ ] Filtrirati signal u smislu spektra signala
- [ ] Podela signala na epohe (od par sekundi sa potencijalnim preklapanjem tipa 50% overlapa)

# Feature extraction nad epohom

> [!info] Pozeljan nacin da se ovo implementira je da postoji funkcija koja primi 1xN ili 62xN numpy niz/matricu i vrati neku vrednost (i ostale parametre koje mozemo da variramo za taj featur). Bice posle lakse kada bude trebalo da se prosledjuje klasifikatoru. Idealno ova funkcija moze da primi MxKxN 3d matricu (tenzor) jer je to neki generalno standardan nacin na koje se stvari prosledjuju u neuralne mreze. Ge je M - broj epoha koje odjednom prosledjujemo (batch size), K - broj kanal najverovatnije 62 i N broj odbiraka. I ovo moze da radi tako ako se npr koristi mean samo se stavim np.mean(niz, axis=-1) 

## Vremenski featuri
- [ ] Nalazenje peakova i peak to peak srednje vreme - **Teodora** - pocela da radi
- [ ] Srednja kvadratna vrednost
- [ ] Varijansa signala
- [ ] [Hjorth Paramteter: Activity](https://en.wikipedia.org/wiki/Hjorth_parameters)
- [ ] [Hjorth Paramteter: Mobility](https://en.wikipedia.org/wiki/Hjorth_parameters)
- [ ] [Hjorth Paramteter: Complexity](https://en.wikipedia.org/wiki/Hjorth_parameters)
## Frekvencijski featuri 
- [ ] Racunanje spektra signala blackmantukey ili welchova metoda
- [ ] Frekvencija na kojoj je snaga maksimalna u spektru
- [ ] Magnituda maksimalne snage
- [ ] Suma spektra
- [ ] Zadnje tri stvari mozda ima smisla raditi posebno za alfa, beta, gama, delta ... kanale - treba proveriti

## Nelinearni dinamicki featuri sistema

ok ovo je lista iz ref rada mozemo da pogledamo par od njih msm da ne mora sve. Imaju reference na radove gde su ove stvari definisane u ref radu. Ali nadam se da imaju u mne.
- [ ] Approximate Entropy
- [ ] C0 Complexity
- [ ] Correlation dimension
- [ ] Kolmogorov Entropy
- [ ] Lyapunov Exponent
- [ ] Permutation entropy
- [ ] Singular entropy
- [ ] Shannon Entropy
- [ ] Spectral entropy

# Adapter sloj
- [ ] Potreban je sloj koji treba da ili ima neki generator koji moze da vrati prozivoljan element za dati indeks jer ne mozemo ceo dataset da stavimo u ram (sto se generalno tice ovog problema kada bude trebalo ceo dataset imam zbog prakse pristup nekim ludim kompovima koji ovo mogu da progutaju bez problema ali to kada bude sve na kraju radilo pa da generismo rezulatet)

# Redukcija dimenzionalnosti?
Jbg nisam seo da radim som i onda nisam sigran kako ovo radi ali bi trebalo baciti pogled na to ali svkakoa treba da se izgenerise koliko je bitan koji featur treba videti kako se to radi (pogledati som)
# Klasifikacija emocija na osnovu izvucenih featura
> [!tip] e sad dok sam pisao ovo sam shvatio da imamo dosta podataka i da sklearn stvari uglavnom zahtevaju da se proslede svi podaci odjendom. To mozda bude problem zbog kolicine podataka ali nisam siguran. Svakako bi adapter sloj trebao da resi to. U slucaju da je to problem onda ce biti lakse samo fallback na neuralne mreze ili videti da li algoritmi u sklean imaju partial_fit funkciju

> [!info] Pre nego sto se uradi ovo treba povezati labele emocija sa epohama ali ovo zapravo treba odraditi u dataloadingu





Za pocetak mozemo da koristimo njihove feature da napravimo klasifikatore pa kad nasi featuri budu gotovi switchujemo se na nase feature
- [ ] Neuralna mrezea
	E sad za ovo mozemo da probamo na kraju msm ako imamo feature koje rade onda ce ovo samo da radi najverovatnije malo bolje. Ovo imam gotovo sa nekih prethodnih projekata tako da moze samo to potencijalno da se iskoristi
- [ ] K-nearest-neighbors sklearn ima ovo implementirano
	I ovo je kao mozda najlakse jer ce da radi odmah za vise klasa koje mi imamo
- [ ] Support vector machine - 
	ovo u principu moze da se uveze samo iz sklearn (scikit-learn puno ime bibliteke) problematika je posto imamo 4 klase kako to da sredimo msm da ima generalno dva pristupa jedan je da imamo da klasifikuje jednu emociju i sve ostalo i onda tako za svaku emociju. Nisam siguran sta je drugi pristup zab sam 

# Rezulati
- [ ] TODO


# Predlog za rad 1
- Moj predlog je da Teodora posto je krenula ove vremenske feature odradi te.
- Sofija moze za sad mozda ove u spektru
- I da svako od nas uzme jedan ili dva ove iz druge grupe featura i da proba da vidi da li negde imaju gotovi ili da ih implemntira ako nisu mnogo teski
- Ja cu dodati onaj prvi deo za data loading jer sam to skoro pisao pa imam u glavi kako moze da se ispise lepo. 
- Za ostatak onda mozemo da se dogovoramo msm ako neko uzme nesto radi moze samo napise na wa i da smatramo da je to zauzet task
# Predlog za rad 2 
- Teodora da radi sve feature
- Sofija da krene odmah sa klasifikatorima 
- Ja cu da radim slojeve koji spajaju stvari
## Pros
- Svi prodjemo po deo svake etape
## Cons
- Relativno je sporije je smo vise ovisni jedni o drugima

## Pros
- Decouplovani smo
- Imamo napredak na par mesta odjednom
## Cons
- Niko nece proci ceo "pipeline" vec samo deo
- Moze da bude sporije ako ja ne procenim dobro te adapterske slojeve