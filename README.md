# ML/DL for Astrophysics — A Practice Project

## Purpose

This repository is a **self-directed learning project**, not original research. It exists to practice applying machine learning and deep learning methods to a real astrophysical problem — spectroscopic redshift estimation from galaxy photometry, structural parameters, and images — as preparation before starting a PhD.

The scientific problem, dataset, and modeling framing are **not my own**. They are adopted directly from a published paper (cited below) so that I could work with a real, well-documented astronomical dataset while learning ML/DL, rather than inventing an artificial problem or using a generic Kaggle dataset with no physical meaning.

## Data & Attribution

All data used in this repository comes from **GalaxiesML**, a public machine-learning-ready dataset of galaxy images, photometry, spectroscopic redshifts, and structural parameters, built from the Hyper-Suprime-Cam Survey PDR2.

**Citation:**

> Do, T., Boscoe, B., Jones, E., Li, Y. Q., & Alfaro, K. (2024). *GalaxiesML: a dataset of galaxy images, photometry, redshifts, and structural parameters for machine learning*. arXiv:2410.00271. https://arxiv.org/abs/2410.00271

This is **not my paper, my dataset, or my original research problem**. All credit for the dataset construction, the underlying science, and the original example redshift-estimation results belongs to the authors above. I am using their public dataset purely as a training ground to practice ML/DL techniques on real astronomical data.

## What This Repository Is

- A structured, self-paced series of experiments that progressively apply ML/DL to the GalaxiesML redshift-estimation problem
- A way to practice moving from tabular photometry → added structural/morphological features → model comparison and interpretation → image-based deep learning (CNNs)
- A demonstration, for my own learning and for PhD applications, that I can independently take a real scientific dataset and apply appropriate ML/DL methodology to it

## What This Repository Is Not

- Not a claim of novel scientific results or a contribution to the redshift-estimation literature
- Not an attempt to reproduce or improve on the original paper's benchmark results in any rigorous, publishable sense
- Not implying the dataset, problem framing, or scientific motivation originated with me

## Structure

- `Experiment_1/` — redshift estimation from photometry alone (baseline)
- `Experiment_2/` — redshift estimation from photometry + structural/morphological parameters
- `Experiment_3/` — broader model comparison and interpretation (planned)
- `Experiment_4/` — image-based deep learning with CNNs (planned)

Each experiment folder contains its own notebook and README with methodology, results, and findings specific to that stage.
