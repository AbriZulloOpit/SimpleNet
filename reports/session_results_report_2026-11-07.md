# Report di sessione: SimpleNet su X-Ray Garments

**Data:** 7 novembre 2026  
**Progetto:** SimpleNet / X-Ray Garments  
**Scopo:** ricostruire il flusso di preparazione dei dati, interpretare i risultati a `imagesize` 288 e 512, e chiarire se la prova a 512 costituisce un nuovo training.

## Sintesi

La struttura del dataset e l’associazione immagine-maschera risultano coerenti con il loader MVTec utilizzato da SimpleNet. Le maschere sono state annotate in CVAT sulle immagini già ridimensionate e padded; non è stato eseguito il resize script sulle maschere, quindi il padding non è stato marcato automaticamente come difetto.

Il confronto tra i CSV del run `xray_garments_fullframe_v1` a 288 e 512 mostra metriche inferiori a 512. Tuttavia, **non è dimostrato che a 512 sia stato allenato un nuovo modello**. I due script usano lo stesso `run_name`; `fit()` carica e riusa `ckpt.pth` se già presente, saltando l’allenamento. Il checkpoint del run risulta modificato il 29 ottobre 2026, mentre i CSV sono stati scritti il 5 novembre 2026. La copia del checkpoint usata dallo script di test ha lo stesso hash SHA-256 del checkpoint del run. Pertanto, il risultato 512 va interpretato come valutazione a risoluzione 512 di un checkpoint preesistente, non come confronto controllato tra due modelli addestrati rispettivamente a 288 e 512.

## Ambito e provenienza delle evidenze

Il report combina:

- Ispezione del codice SimpleNet attualmente presente nel workspace: script train/test, loader MVTec, `fit()`, `test()`, calcolo e salvataggio delle metriche.
- CSV correnti presenti nel repository per le run a 512 e per la vecchia run `xray_garments_run`.
- Valori del CSV della run full-frame a 288 forniti dall’utente in conversazione; quel CSV storico è stato poi sovrascritto dalla run a 512 e non è attualmente presente nella directory di output.
- Conteggi e verifiche del dataset eseguiti durante la sessione: conteggio file, corrispondenza dei nomi immagine-maschera e assenza di sovrapposizioni di nome tra `train/good` e `test/good`.
- Percorsi, script e valori descritti o allegati dall’utente. L’assistente non ha eseguito training, valutazione o preprocessing del dataset.

## Dati e protocollo

### Preparazione e layout

Le immagini della cartella originale sono state convertite in un canvas quadrato di 1024 px preservando l’aspect ratio e aggiungendo padding (`--pad --fill edge`) tramite lo script PowerShell e la CLI del progetto [run_resize.ps1](../../ml-pin-detector/scripts/run_resize.ps1) e [resize_images.py](../../ml-pin-detector/scripts/resize_images.py). Secondo quanto riferito dall’utente, le maschere sono state successivamente annotate con CVAT direttamente sulle immagini resized/padded; il resize script non è stato applicato alle maschere.

Il layout usato è quello atteso dal loader MVTec di SimpleNet, in [datasets/mvtec.py](../datasets/mvtec.py): `xray-garments/train/good`, `xray-garments/test/<classe>` e `xray-garments/ground_truth/<classe>`.

| Cartella / classe | Immagini | Maschere | Verifica |
|---|---:|---:|---|
| `train/good` | 500 | non previste | 0 nomi uguali a `test/good` |
| `test/good` | 421 | non previste | normale |
| `test/mixed` | 1 | 1 | nessuna maschera mancante/orfana |
| `test/paperclip` | 15 | 15 | nessuna maschera mancante/orfana |
| `test/rivet` | 16 | 16 | nessuna maschera mancante/orfana |
| `test/screw` | 9 | 9 | nessuna maschera mancante/orfana |
| **Totale test** | **462** | **41** | **41 immagini anomale annotate** |

Le verifiche hanno trovato 0 nomi immagine-maschera discordanti e 0 maschere mancanti o orfane. I file sono allineati per nome. Il loader, tuttavia, costruisce le coppie per ordinamento e indice; controllare i nomi resta utile perché il codice non fa un join esplicito per filename.

Esempio controllato dopo il resize interno del loader: immagine `test/mixed/00010_VESTITI___1764687649_583_R.png`, maschera con lo stesso nome in `ground_truth/mixed`; tensori rispettivamente `[3, 288, 288]` e `[1, 288, 288]`, con 25 pixel positivi nella maschera trasformata. Questi sono circa lo 0,030% di 288 × 288 pixel. Il dato indica una ground truth molto sottile a questa risoluzione, ma non dimostra che l’annotazione sia errata o che il resize sia la causa delle metriche.

Le maschere vengono lette come grayscale, ridimensionate con interpolazione nearest-neighbor e binarizzate come `mask > 0`. Per una validazione visiva corretta, l’overlay va generato dopo le trasformazioni effettive del loader e verificato anche a 512.

### Configurazione del modello

Gli script full-frame usano seed 0, backbone WideResNet50, feature layer2/layer3, dimensioni di embedding pre-train e target 1536, patch size 3, 5 meta-epoche, 4 epoche GAN, batch size 2, rumore 0,015, discriminatore hidden 1024/layer 2/margin 0,5 e `pre_proj=1`. La differenza esplicita fra le versioni full-frame fornite è `imagesize`: 288 nella run storica e 512 nella run corrente. Lo script corrente di training è [run_xray_garments_fullframe_v1.sh](../run_xray_garments_fullframe_v1.sh); quello di valutazione è [test_xray_garments_fullframe_v1.sh](../test_xray_garments_fullframe_v1.sh).

Il loader SimpleNet attuale ridimensiona direttamente l’intero canvas quadrato a `imagesize × imagesize`, senza crop. Ridimensiona anche la maschera alla stessa dimensione con nearest-neighbor. Il parametro CLI `--resize` è mantenuto per compatibilità e non controlla il resize effettivo in questa versione del loader.

### Flusso di training/evaluation

Senza `--test`, [main.py](../main.py) chiama `SimpleNet.fit(training, testing)`. Il secondo loader viene valutato a ogni meta-epoca e le sue metriche sono usate per selezionare il checkpoint migliore. Il set chiamato `testing` è quindi anche set di selezione del modello.

Con `--test`, `main.py` chiama `SimpleNet.test(...)`; questo carica il checkpoint nella directory del run di test e calcola le metriche sul loader passato. Il test script copia il checkpoint dal run di training. Perciò i CSV di training/evaluation coincidono quando il checkpoint e il set di valutazione sono gli stessi. Non sono due valutazioni indipendenti e il test attuale non fornisce una stima finale non coinvolta nella selezione del modello.

## Risultati

Le metriche sono AUROC in scala 0–1. I CSV hanno una colonna ambigua: `anomaly_pixel_auroc` contiene in realtà **PRO**, perché `_evaluate()` restituisce `auroc, full_pixel_auroc, pro`, e `main.py` assegna il terzo valore a quella chiave. La metrica pixel AUROC riportata in tabella qui sotto è quindi `full_pixel_auroc`.

| Run / `imagesize` | Instance AUROC | Full Pixel AUROC | PRO (colonna CSV `anomaly_pixel_auroc`) | Fonte / stato |
|---|---:|---:|---:|---|
| `xray_garments_fullframe_v1`, 288 | 0,977580 | 0,084843 | 0,000000 | CSV storico fornito dall’utente; non più presente perché sovrascritto |
| `xray_garments_fullframe_v1`, 512 | 0,886623 | 0,014870 | 0,000000 | CSV attuale presente nel repository |
| Differenza 512 − 288 | −0,090956 | −0,069973 | 0,000000 | Confronto di output, ma non di due training verificati |
| `xray_garments_run`, 288* | 0,922658 | 0,853544 | 0,645463 | Ulteriori CSV storici presenti nel repository; run distinta |

`*` Il repository contiene anche CSV della vecchia run `xray_garments_run` e della relativa evaluation, con valori identici. Questa coppia non va fusa con il confronto full-frame: il suo CSV storico non documenta nel contenuto quale esatta configurazione lo abbia prodotto. Inoltre, lo script [run_xray_garments.sh](../run_xray_garments.sh) attualmente presente usa `run_name=xray_garments_dullframe_v1`, mentre [test_xray_garments.sh](../test_xray_garments.sh) punta a `trained_run=xray_garments_run`. Prima di usare quei valori come baseline occorre identificare il comando/checkpoint originale. In ogni caso la differenza di AUROC pixel è troppo ampia per attribuirla soltanto al passaggio da 288 a 512.

### Interpretazione

Nel confronto full-frame, instance AUROC è diminuita di circa 9,10 punti percentuali; full-pixel AUROC è diminuita di circa 7,00 punti; PRO è zero in entrambe le versioni. Il modello individua quindi le immagini anomale meglio di quanto localizzi i pixel, mentre la performance di localizzazione risulta molto debole in entrambi i CSV full-frame.

Non si può concludere che un modello **addestrato a 512** sia peggiore di uno addestrato a 288. Il checkpoint full-frame esistente ha timestamp 2026-10-29 13:23, mentre i CSV full-frame sono del 2026-11-05 17:33. Il sorgente e la copia del checkpoint per il test hanno dimensione uguale e hash SHA-256 uguale. `fit()` carica un checkpoint esistente e ritorna dopo la valutazione, senza eseguire il ciclo di training né risalvare i pesi. In base al precedente run a 288 con lo stesso `run_name`, è fortemente indicato che il comando impostato a 512 abbia riusato il checkpoint precedente; i risultati misurano quindi l’effetto dell’inferenza a risoluzione 512 su quel checkpoint, non un nuovo training a 512.

I CSV di training e test riportano identici valori in entrambe le risoluzioni. Ciò è coerente con il flusso attuale: `fit()` usa il set `test` anche per scegliere il checkpoint, quindi il CSV scritto al training riflette le metriche su quel set; lo script `--test` rivaluta il checkpoint sugli stessi esempi. La coincidenza non è un’ulteriore conferma indipendente delle prestazioni.

### Risultati visivi e localizzazione

Le mappe mostrate in sessione appaiono ampie e sfumate rispetto alle annotazioni sottili. Una heatmap è una mappa di score, non una segmentazione binaria; il colore caldo segnala score più alto e non significa da solo che ogni pixel circostante sia etichettato come difetto.

La maschera CVAT di esempio è nera sullo sfondo e contiene pochi tratti colorati; dato che il loader converte a grayscale e considera ogni valore maggiore di zero positivo, i tratti colorati contribuiscono alla ground truth, mentre il nero resta background. Il padding bianco non è stato annotato come difetto perché le maschere sono state create successivamente sulle immagini già padded. Il padding può comunque modificare la distribuzione visiva delle immagini e occupa una porzione del canvas, ma non è dimostrato come causa delle mappe diffuse. La bassa pixel AUROC/PRO e il supporto di soli 25 pixel nella maschera trasformata sono segnali da indagare con overlay e visualizzazione, non una diagnosi causale.

## Limiti e minacce alla validità

- **Riutilizzo del checkpoint:** la prova a 512 non risulta un nuovo training. Una comparazione di risoluzione richiede checkpoint distinti e verificati.
- **Selezione sul test:** il set chiamato `test` è usato durante `fit()` per selezionare il checkpoint. Le metriche dei CSV non sono una stima indipendente di generalizzazione.
- **Campione anomalo limitato:** 41 anomalie totali, con una sola immagine `mixed`. Le metriche per classe e le divisioni validation/test sono instabili o non possibili per tutte le classi.
- **Annotazioni piccole:** nel caso di esempio la maschera trasformata ha 25 pixel positivi a 288. Il numero di pixel positivi prima del resize e a 512 non è stato verificato.
- **Run storica aggiuntiva non riconciliata:** i risultati `xray_garments_run` sono molto diversi dai risultati full-frame e i nomi/configurazioni degli script presenti non identificano con certezza quale invocazione li abbia generati.
- **Nessun training rieseguito durante questa sessione:** le conclusioni quantitative derivano dai risultati forniti e dai file di output; l’assistente ha effettuato ispezioni di codice e verifiche file/dataset, non lanciato SimpleNet.
- **Riproducibilità limitata:** i run specifici mostrati usano seed 0 e una configurazione breve (5 meta-epoche); non sono disponibili repliche con seed diversi o deviazioni standard.

## Conclusioni

1. Il layout dataset è corretto per il loader MVTec; i nomi delle 41 maschere combaciano con quelli delle immagini e il train/test dei `good` non presenta nomi duplicati.
2. Il padding non è una falsa annotazione nelle maschere nel flusso descritto, perché CVAT è stato usato dopo il resize/padding. Resta possibile un impatto indiretto del fondo e del basso supporto delle maschere, non dimostrato dai dati correnti.
3. I punteggi full-frame a 512 sono peggiori di quelli a 288 nei CSV, ma la run a 512 sembra aver riutilizzato i pesi precedenti. Non costituisce pertanto una prova che il training a 512 peggiori SimpleNet.
4. La localizzazione è il problema principale: pixel AUROC è 0,084843 a 288 e 0,014870 nell’inferenza 512 attuale; PRO è 0 in entrambi i casi. La colonna del CSV chiamata `anomaly_pixel_auroc` non rappresenta anomaly pixel AUROC, ma PRO.

## Passi successivi raccomandati

1. **Eseguire un vero training a 512:** assegnare un run name nuovo, ad esempio `xray_garments_fullframe_512_fresh_v1`, e verificare che la sua cartella `models/0/mvtec_xray-garments/ckpt.pth` non esista prima dell’esecuzione. Non affidarsi alla modalità `overwrite`: crea/riusa la directory ma non ne cancella il checkpoint.
2. Dopo il comando, controllare che il checkpoint sia stato scritto con timestamp nuovo, che il log mostri il training e che la evaluation copi quel checkpoint. Mantenere `imagesize=512` e la stessa configurazione in entrambi gli script.
3. **Separare selezione e test finale senza modificare SimpleNet upstream:** creare una cartella di validazione nella quale il loader legga le immagini di validation dal suo split `test/`, con `ground_truth` corrispondenti; puntare il training a quella cartella. Puntare poi lo script `--test` a una cartella separata di test finale. Replicare `train/good` nelle due strutture se serve al loader. Usare un run name nuovo per il training.
4. Non dividere la singola immagine `mixed` tra validation e test: occorrono altre immagini di questa categoria per una metrica indipendente. Se le immagini appartengono allo stesso capo/sequenza/acquisizione, mantenerle nello stesso split.
5. Salvare per le stesse immagini overlay di image + ground truth + heatmap, sia a 288 sia a 512, e misurare `mask.sum()` a entrambe le risoluzioni. Questo distinguerà perdita di pixel, errore di mappatura e scarsa localizzazione del modello.
6. Per una valutazione più robusta, raccogliere nuove immagini anomale, includere più maschere per ogni classe e ripetere i run con più seed.

## Artefatti di riferimento

- [Script di training full-frame](../run_xray_garments_fullframe_v1.sh)
- [Script di evaluation full-frame](../test_xray_garments_fullframe_v1.sh)
- [Risultati full-frame di training, CSV attuale](../results/XrayGarments_Results/simplenet_xray_garments/xray_garments_fullframe_v1/results.csv)
- [Risultati full-frame di test, CSV attuale](../results/XrayGarments_Results/simplenet_xray_garments/xray_garments_fullframe_v1_test/results.csv)
- [Loader MVTec](../datasets/mvtec.py)
- [Implementazione SimpleNet](../simplenet.py)
- [Metriche](../metrics.py)
- [Script di resize](../../ml-pin-detector/scripts/run_resize.ps1)
- [CLI di resize](../../ml-pin-detector/scripts/resize_images.py)
