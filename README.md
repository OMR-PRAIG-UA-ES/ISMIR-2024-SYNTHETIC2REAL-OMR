<p align='center'>
  <a href='https://praig.ua.es/'><img src='https://i.imgur.com/Iu7CvC1.png' alt='PRAIG-logo' width='100'></a>
</p>

<h1 align='center'>Synthetic-to-Real Domain Adaptation for Optical Music Recognition</h1>

<!---
<h4 align='center'>Full text coming soon<a href='' target='_blank'></a>.</h4>
--->

<p align='center'>
  <img src='https://img.shields.io/badge/python-3.9.0-orange' alt='Python'>
  <img src='https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white' alt='PyTorch'>
  <img src='https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white' alt='Lightning'>
  <img src='https://img.shields.io/static/v1?label=License&message=MIT&color=blue' alt='License'>
</p>

<p align='center'>
  <!---<a href='#about'>About</a> •--->
  <a href='#how-to-use'>How To Use</a> •
  <!---<a href='#citations'>Citations</a> •
  <a href='#acknowledgments'>Acknowledgments</a> •--->
  <a href='#license'>License</a>
</p>

<!---
## About
--->


## How To Use

### Set up

You can use the included [`Dockerfile`](Dockerfile):
```bash
docker build --tag omr_amd:latest .
```

### Datasets

**We evaluate our approach on mensural music notation:**
- We use *Capitan* (or b-59-850), *Il Lauro Secco*, *Magnificat*, *Mottecta*, and *Guatemala* datasets. These are private datasets and are available upon [request](mailto:malfaro@dlsi.ua.es). After obtaining these datasets, please place them in the [`data`](data) folder.


### Experiments

We use Primens as the source dataset and try to adapt its corresponding source model to each of the remaining datasets.
There are two adaptation stages:
1. Source-Free Domain Adaptation using the AMD method
  We perform a random seach of 50 runs for each source-target combination and keep the best one as the final result.
2. Source-Free Self-Labelling Adaptation
  The best model from 1. is further fine-tuned using a self-labelled dataset. Such a dataset is first created using the shelf-off predictions from model 1. based on a confidence threshold. That first dataset is used to fine-tune in a supervised way the model. At the end of each epoch, the remaining samples (those not considered due to low confidence) are evaluated again to see if they can be now considered.


Execute the [`run_experiments.sh`](run_experiments.sh) script to replicate the experiments from our work:
```bash 
$ bash run_experiments.sh
```


<!---
## Citations

```bibtex
@inproceedings{,
  title     = {{}},
  author    = {},
  booktitle = {{}},
  year      = {},
  publisher = {},
  address   = {},
  month     = {},
}
```



## Acknowledgments

This work is part of the I+D+i PID2020-118447RA-I00 ([MultiScore](https://sites.google.com/view/multiscore-project)) project, funded by MCIN/AEI/10.13039/501100011033.
--->
## License

This work is under a [MIT](LICENSE) license.
