# Variational Auto Encoder on Mnist Data Set



### Dependencies


* python 3.5+
* tensorflow >= 2.0
* sklearn


### Executing program

Define number of epochs, batch size and latent dimension

```
python3 vae.py --epochs 10 --batch_size 128 --latent_dim 100
```
In order to load a model with pretrained weights

```
python3 vae.py --load True
```

## Authors

Allon Hammer