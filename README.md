# PLM_AAV

### Summary

This repo contains code relating to a side project looking at the ability of ML models to learn the sequence -> function relationship within a dataset from **Bryant, D. H., Bashir, A., Sinai, S., Jain, N. K., Ogden, P. J., Riley, P. F., ... & Kelsic, E. D. (2021). Deep diversification of an AAV capsid protein by machine learning. *Nature Biotechnology*, *39*(6), 691-696.**

### Scripts

#### generate_dataset.py

* Downloads dataset from Bryant, Bashir [...] & Kelsic supp table
* Formats it for intake by other scripts

#### run_zero_shot.py

* Applies a zero-shot PLM based model to the datasets, assesses how well the zero-shot prediction of functionality predicts actual performance
* Also, caches representations of the datasets extracted from the ESM2 
* **Will cache ~320gb of tensors on disk**

#### run_optuna.py

* Uses optuna to search hyper-parameter space for 5 supervised models that try to learn sequence-function relationship on a similar data split as used in Bryant, Bashir [...] & Kelsic
* Trains multiple replicates using best hyper-parameters, tests their performance on the model-designed sequences from Bryant, Bashir [...] & Kelsic which were trained on the same dataset
