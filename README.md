# TVShowRecommender

##Requisiti
* Julia v. 0.4 - [http://julialang.org/downloads/](http://julialang.org/downloads/) sezione "Nightly builds"
* Se Juno non trova il package "Jewel" aprire la console di Julia e installare il pacchetto:
```
Pkg.clone("git://github.com/one-more-minute/Jewel.jl.git", "Jewel")
```

##Creazione liste per evaluation
Il tool 'data_extraction_tool' permette la creazione delle liste 'training.txt' e 'testing.txt' necessarie per l'esecuzione dell'algoritmo.
Il tool rimuove i programmi duplicati e crea gli elenchi mantenendo una percentuale tra il numero di programmi di training e di testing.
Il tool presenta ancora un problema riguardo la codifica dei numeri, controllare la lista alla ricerca di valori esponenziali

##Evaluation
Per l'evaluation caricare nella cartella 'dataset' il file 'auditel.txt' e rinominarlo in 'data.txt'
Non committare questo file!