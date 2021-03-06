# Lab 3 of ML Cyber Security

```bash
├── original_data: from https://github.com/csaw-hackml/CSAW-HackML-2020, 
                   containing some data and models needed inside the jupyter notebook
├── models
    └── bd_net.h5
    └── bd_weights.h5
    └── mask_2.npy
    └── mask_4.npy
    └── mask_10.npy
    └── mask_30.npy
├── eval_2.py
├── eval_4.py
├── eval_10.py
├── eval_30.py
├── Lab3_report.pdf
├── plot.png
└── repair.ipynb
```

### Some instructions

To evaluate the model after pruning, execute one of the evaluation scripts by running:  
`python3 eval_n.py <clean validation data directory> <poisoned validation data directory>`, where `n` is the accuracy drop in validation set.

E.g., `python3 eval_2.py data/cl/valid.h5 data/bd/bd_valid.h5`, if your data is at the right position.

The models are generated by the `repair.ipynb` file, which you can look into. If you want to run the notebook, you may need to adjust the file positions or change the position definition in the notebook

The `plot.png` is the plot of the accuracy on clean test data and the attack success rate (on backdoored test data) as a function of the fraction of channels pruned. It is also described in the report.