# Dove va il tempo del DNS

Misura del 16 settembre 2026. **Le trasformate consumano circa il 66–67%
 dello step; i prodotti convettivi un altro 14–15%.** Il primo bersaglio resta
il termine nonlineare, ma il broadcasting merita un intervento specifico.

Non sono state modificate le routine del solutore durante questo profiling.
I numeri completi, gli hash delle sorgenti e i tempi dei kernel sono in
[profiling-details.md](profiling-details.md). Lo stato misurato contiene
modifiche non committate: il solo identificativo `be56ce2` non lo identifica.

## Condizioni e attendibilità

- Couette, Re=400; Ny=35, Nx=Nz=32; griglia fisica padded 35×48×48.
- CNRK2, dt=0.025, forma convettiva, nessun forcing aggiuntivo e nessun monitor.
- Julia 1.12.6, Apple M1; un thread Julia, BLAS e FFTW.
- Pianificazione, inizializzazione e compilazione escluse dai tempi.
- 21 campioni a caldo per misura; stato iniziale ripristinato prima di ogni
  step. RNG MersenneTwister, seed 42, rumore iniziale di scala 0.01.
- Le fasi sono cronometrate in intervalli disgiunti, sommati sui tre stadi.
  La copia strumentata dello step coincide con lo step di produzione:
  differenza massima **0.0** sia nella velocità sia nella pressione.
- Un secondo controllo usa il profiler a campionamento su 100 step di
  produzione, includendo i frame nativi di FFTW.

La macchina mostra variabilità apprezzabile: lo step di produzione ha mediana
47.12 ms con ESTIMATE e 50.94 ms con MEASURE, ma minimi rispettivamente
46.05 e 44.82 ms. Alcuni campioni arrivano a 88–100 ms. **Queste mediane
non dimostrano che ESTIMATE sia più veloce di MEASURE.** Le misure sono
sequenziali e risentono del carico e della frequenza CPU.

Lo step strumentato ha media 64.75 ms con ESTIMATE e 45.61 ms con MEASURE.
Questo scarto rispetto alle misure di produzione impedisce di trasferire
meccanicamente i millisecondi da una tabella all'altra. Le quote delle fasi
sono però molto simili nelle due sessioni: sono l'indicazione più utile
per decidere dove intervenire. Non sono intervalli di confidenza statistici.

## Ripartizione dello step

Qui i millisecondi sono le **medie dello step strumentato MEASURE**;
le percentuali ESTIMATE sono riportate per confronto. Le righe sono disgiunte.

| Fase, su tutti e tre gli stadi | ms/step MEASURE | Quota MEASURE | Quota ESTIMATE |
|---|---:|---:|---:|
| Trasformate inverse dei gradienti | 18.190 | 39.88% | 39.94% |
| Prodotti convettivi in spazio fisico | 6.920 | 15.17% | 13.67% |
| Trasformate inverse della velocità | 6.101 | 13.38% | 13.47% |
| Trasformate dirette del termine nonlineare | 5.698 | 12.49% | 14.05% |
| Solve di Stokes / influence matrix | 4.998 | 10.96% | 10.49% |
| Assemblaggio CN: Laplaciani, gradiente di pressione, combinazioni | 1.698 | 3.72% | 3.66% |
| Derivate spettrali della velocità | 0.701 | 1.54% | 1.59% |
| Aggiornamento storia RK | 0.529 | 1.16% | 1.23% |
| Copia velocità e aggiunta base flow | 0.388 | 0.85% | 0.98% |
| Scrittura RHS nonlineare | 0.383 | 0.84% | 0.90% |
| Controllo e strumentazione fuori dagli intervalli | 0.005 | 0.01% | 0.03% |

Nel complesso, l'intera valutazione nonlineare prende circa **84%** dello step.
Le sole tre famiglie di trasformate prendono **65.75%** nella misura MEASURE.

## Perché le trasformate pesano così tanto

Ogni stadio calcola tre componenti di velocità e nove componenti del gradiente
in spazio fisico, poi trasforma le tre componenti del prodotto convettivo.
Quindi ogni step esegue:

- **36 trasformate inverse scalari**: 9 per la velocità e 27 per i gradienti;
- **9 trasformate dirette scalari**.

Ogni trasformata scalare comprende una parte Fourier e una Chebyshev, oltre
a copie, padding/troncamento e normalizzazione. La ripetizione amplifica
anche costi inferiori al millisecondo.

Tempi isolati MEASURE, mediane per chiamata:

| Kernel | ms/chiamata |
|---|---:|
| Inversa completa | 0.6486 |
| DCT-I inversa sui modi risolti | 0.3679 |
| Fourier inversa | 0.1985 |
| Diretta completa | 0.6020 |
| DCT-I diretta sui modi risolti | 0.3597 |
| Fourier diretta | 0.1701 |

La DCT è il kernel più costoso dentro entrambe le trasformate. **Questi tempi
isolati non vanno aggiunti ai tempi delle fasi:** sono misure annidate,
con condizioni di cache diverse. Anche la differenza tra tempo completo e
somma dei kernel non è una misura esatta delle copie.

## Il secondo bersaglio: broadcasting dei prodotti convettivi

`dot!` in [gradientfield.jl](../src/fields/gradientfield.jl) costruisce, per
ogni componente, tre prodotti e due somme punto per punto. Il costo misurato
è 6.92 ms per step, pur senza allocazioni Julia nello step completo.

Il profilo di produzione mostra campioni significativi in `getindex` del
broadcast, `combine_axes`, `broadcast_shape` e `_bcs1`. Il percorso passa
per il `materialize!` personalizzato dei campi: il costo non è soltanto
quello delle moltiplicazioni numeriche.

**Esperimento prioritario:** confrontare questa operazione con broadcast
sugli array `parent` e con un ciclo lineare specializzato. Prima verificare
l'identità del risultato, poi misurare. Il profiling indica un candidato
concreto, ma non dimostra ancora quanto tempo si recupererebbe modificandolo.

## Solve e operatori

I tre solve di Stokes prendono insieme circa 5 ms, cioè 11%. Sono rilevanti,
ma anche eliminarne interamente il costo darebbe soltanto circa **1.12×**
di accelerazione complessiva secondo questa ripartizione.

Le derivate spettrali non sono il problema principale: il gradiente completo
della velocità pesa circa 1.5%; il restante assemblaggio CN circa 3.7%.
Per confronto, una chiamata isolata al Laplaciano costa 0.0366 ms e una
singola derivata 0.022–0.031 ms. Ci sono nove Laplaciani e dodici chiamate
per ciascuna direzione di derivazione per step.

## Allocazioni e limiti

Lo step di produzione continua a misurare **zero byte allocati nell'heap
Julia**. Questo non significa zero allocazioni native: nel profilo compaiono
`fftw_malloc_plain`, malloc e free. Non abbiamo misurato il totale di byte
allocati internamente da FFTW.

Nel test isolato ESTIMATE alcuni wrapper dei kernel riportano 16 byte;
questo non compare nello step completo né nella corrispondente misura
MEASURE. Non va interpretato come un'allocazione dimostrata per ciascuna
trasformata nel percorso di produzione.

## Ordine consigliato per il prossimo intervento

1. **Prodotti convettivi e broadcast dei campi.** Circa 15% del costo;
   esperimento piccolo, senza cambiare formulazione numerica.
2. **DCT e numero di trasformate.** È il margine maggiore. Valutare kernel
   alternativi e organizzazione delle trasformate a risoluzione invariata.
   La forma rotativa richiede meno inverse, ma il confronto deve includere
   curl, inizializzazione della pressione modificata e verifica numerica:
   non è una sostituzione puramente implementativa.
3. **Solve modali.** Ottimizzarli dopo i due punti precedenti, misurando
   separatamente Helmholtz, derivate dei coefficienti e correzione influence.
4. **Micro-ottimizzazioni delle derivate.** Priorità bassa nel caso attuale.

Come limite ideale, dimezzare il costo di tutte le trasformate, lasciando
il resto invariato, darebbe circa **1.49×** di accelerazione totale.
Sono limiti di Amdahl basati sulle quote misurate, non promesse di speedup.

## Riproduzione

Dalla root del package:

```sh
julia --startup-file=no --threads=1 --project=. perf/profile_detailed.jl
```

Lo script rigenera [profiling-details.md](profiling-details.md),
[cpu-detailed-estimate.txt](cpu-detailed-estimate.txt) e
[cpu-detailed-measure.txt](cpu-detailed-measure.txt). Il presente riassunto
è un'interpretazione della misura datata sopra e non viene rigenerato.
Se cambia lo step di produzione, aggiornare anche la sua copia strumentata:
il controllo di equivalenza aiuta a rilevare scostamenti nel caso coperto,
ma non sostituisce la verifica di altre forme, forcing o vincoli di flusso.
