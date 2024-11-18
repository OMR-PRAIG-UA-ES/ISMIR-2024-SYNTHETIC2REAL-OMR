<p align='center'>
  <a href='https://praig.ua.es/'><img src='https://i.imgur.com/Iu7CvC1.png' alt='PRAIG-logo' width='100'></a>
</p>

<h1 align='center'>Synthetic-to-Real Domain Adaptation for Optical Music Recognition</h1>

<h4 align='center'>Full text available <a href='https://ismir2024program.ismir.net/poster_45.html' target='_blank'>here</a>.</h4>

<p align='center'>
  <img src='https://img.shields.io/badge/python-3.10.0-orange' alt='Python'>
  <img src='https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white' alt='PyTorch'>
  <img src='https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white' alt='Lightning'>
  <img src='https://img.shields.io/static/v1?label=License&message=MIT&color=blue' alt='License'>
</p>

<p align='center'>
  <!---<a href='#about'>About</a> •--->
  <a href='#how-to-use'>How To Use</a> •
  <a href='#citations'>Citations</a> •
  <a href='#acknowledgments'>Acknowledgments</a> •
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
- We use *Capitan* (or Zaragoza), *Il Lauro Secco*, *Magnificat*, *Mottecta*, and *Guatemala* datasets. These are private datasets and are available upon [request](mailto:malfaro@dlsi.ua.es). After obtaining these datasets, please place them in the [`data`](data) folder.

### Experiments

We use Primens as the source dataset and try to adapt its corresponding source model to each of the remaining datasets. We use the Align, Minimize and Diversify (AMD) method to perform a source-free domain adaptation. Specifically, we perform a random seach of 50 runs for each source-target combination and keep the best one as the final result.

Execute the [`run_experiments.sh`](run_experiments.sh) script to replicate the experiments from our work:
```bash 
$ bash run_experiments.sh
```


## Citations

```bibtex
@inproceedings{luna2024syn2realomr,
  title     = {{Unsupervised Synthetic-to-Real Adaptation for Optical Music Recognition}},
  author    = {Luna-Barahona, Noelia and Roselló, Adrián and Alfaro-Contreras, Mar{\'\i}a and Rizo, David and Calvo-Zaragoza, Jorge},
  booktitle = {{Proceedings of the 25th International Society for Music Information Retrieval Conference}},
  year      = {2024},
  publisher = {ISMIR},
  address   = {San Francisco, United States},
  month     = {nov},
}
```


## Acknowledgments

This work is supported by grant CISEJI/2023/9 from "Programa para el apoyo a personas investigadoras con talento (Plan GenT) de la Generalitat Valenciana".


## License

This work is under a [MIT](LICENSE) license.
