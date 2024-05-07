# 👂 💉 EHRSHOT

A benchmark/dataset for few-shot evaluation of foundation models for electronic health records (EHRs). You can **[read the paper here](https://arxiv.org/abs/2307.02028)**. 

Please note that the dataset + model are still being reviewed, and a download link will be provided once they are approved for public release.

----

Whereas most prior EHR benchmarks are limited to the ICU setting, **EHRSHOT** contains the **full longitudinal health records of 6,739 patients from Stanford Medicine** and a diverse set of **15 classification tasks** tailored towards few-shot evaluation of pre-trained models. 

# 📖 Table of Contents
1. [Pre-trained Foundation Model](#models)
2. [Dataset + Tasks](#dataset)
3. [Comparison to Prior Work](#prior_work)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Citation](#citation)


<a name="models"/>

# 🔮 Foundation Model for EHRs

**Access:** [The model is on HuggingFace here](https://huggingface.co/StanfordShahLab/clmbr-t-base) and requires signing a research usage agreement.

We publish the model weights of a **141 million parameter** clinical foundation model pre-trained on the deidentified structured EHR data of **2.57M patients** from Stanford Medicine.

We are [one of the first](https://arxiv.org/abs/2303.12961) to fully release such a model for coded EHR data; in contrast, most prior models released for clinical data  (e.g. GatorTron, ClinicalBERT) only work with unstructured text and cannot process the rich, structured data within an EHR.

We use [Clinical Language-Model-Based Representations (CLMBR)](https://www.sciencedirect.com/science/article/pii/S1532046420302653) as our model. CLMBR is an autoregressive model designed to predict the next medical code in a patient's timeline given previous codes. CLMBR employs causally masked local attention, ensuring forward-only flow of information which is vital for prediction tasks and is in contrast to BERT-based models which are bidirectional in nature. We utilize a transformer as our base model with 141 million trainable parameters and a next code prediction objective, providing minute-level EHR resolution rather than the day-level aggregation of the original model formulation. 


<a name="dataset"/>

# 🗃️ Dataset + Tasks

**Access:** [The EHRSHOT-2023 dataset is available on Redivis here](https://redivis.com/ShahLab/editor/datasets/53gc-8rhx41kgt) and requires signing a research usage agreement.

EHRSHOT-2023 contains:
* **6,739 patients**
* **41.6 million clinical events**
* **921,499 visits**
* **15 prediction tasks**

Each patient consists of an ordered timeline of clinical events taken from the structured data of their EHR (e.g. diagnoses, procedures, prescriptions, etc.). 

Each task is a predictive classification task, and includes a canonical train/val/test split. The tasks are defined as follows:

|         Task         | Type              | Prediction Time                       | Time Horizon           |
|:--------------------:|-------------------|---------------------------------------|------------------------|
| Long Length of Stay  | Binary            | 11:59pm on day of admission           | Admission duration     |
| 30-day Readmission   | Binary            | 11:59pm on day of discharge           | 30-days post discharge |
| ICU Transfer         | Binary            | 11:59pm on day of admission           | Admission duration     |
| Thrombocytopenia     | 4-way Multiclass  | Immediately before result is recorded | Next result            |
| Hyperkalemia         | 4-way Multiclass  | Immediately before result is recorded | Next result            |
| Hypoglycemia         | 4-way Multiclass  | Immediately before result is recorded | Next result            |
| Hyponatremia         | 4-way Multiclass  | Immediately before result is recorded | Next result            |
| Anemia               | 4-way Multiclass  | Immediately before result is recorded | Next result            |
| Hypertension         | Binary            | 11:59pm on day of discharge           | 1 year post-discharge  |
| Hyperlipidemia       | Binary            | 11:59pm on day of discharge           | 1 year post-discharge  |
| Pancreatic Cancer    | Binary            | 11:59pm on day of discharge           | 1 year post-discharge  |
| Celiac               | Binary            | 11:59pm on day of discharge           | 1 year post-discharge  |
| Lupus                | Binary            | 11:59pm on day of discharge           | 1 year post-discharge  |
| Acute MI             | Binary            | 11:59pm on day of discharge           | 1 year post-discharge  |
| Chest X-Ray Findings | 14-way Multilabel | 24hrs before report is recorded       | Next report            |



<a name="prior_work"/>

# 📊 Comparison to Prior Work

Most prior benchmarks are (1) limited to the ICU setting and (2) not tailored towards few-shot evaluation of pre-trained models.

In contrast, **EHRSHOT** contains (1) the full breadth of longitudinal data that a health system would expect to have on the patients it treats and (2) a broad range of tasks designed to evaluate models' task adaptation and few-shot capabilities:

<table>
  <tr> <th rowspan="3">Benchmark</th> <th colspan="1">Source</th> <th colspan="3">EHR Properties</th> <th colspan="2">Evaluation</th> <th colspan="3">Reproducibility</th> </tr>
  <tr> <td rowspan="2">Dataset</td> <td rowspan="2">ICU/ED Visits</td> <td rowspan="2">Non-ICU/ED Visits</td> <td rowspan="2"># of Patients</td> <td rowspan="2"># of Tasks</td> <td rowspan="2">Few Shot</td> <td rowspan="2">Dataset via DUA</td> <td rowspan="2">Preprocessing Code</td> <td rowspan="2">Model Weights</td> </tr>
  <tr></tr>
  <tr></tr>
  <tr> <td><b>EHRSHOT</b></td> <td><b>Stanford Medicine</b></td> <td><b>✓</b></td> <td><b>✓</b></td> <td><b>7k</b></td> <td><b>15</b></td> <td><b>✓</b></td> <td><b>✓</b></td> <td><b>✓</b></td> <td><b>✓</b></td> </tr>
  <tr> <td><a href="https://github.com/MLforHealth/MIMIC_Extract">MIMIC-Extract</a></td> <td>MIMIC-III</td> <td>✓</td> <td>--</td> <td>34k</td> <td>5</td> <td>--</td> <td>✓</td> <td>✓</td> <td>--</td> </tr>
  <tr> <td><a href="https://github.com/USC-Melady/Benchmarking_DL_MIMICIII">Purushotham 2018</a></td> <td>MIMIC-III</td> <td>✓</td> <td>--</td> <td>35k</td> <td>3</td> <td>--</td> <td>✓</td> <td>✓</td> <td>--</td> </tr>
  <tr> <td><a href="https://github.com/YerevaNN/mimic3-benchmarks">Harutyunyan 2019</a></td> <td>MIMIC-III</td> <td>✓</td> <td>--</td> <td>33k</td> <td>4</td> <td>--</td> <td>✓</td> <td>✓</td> <td>--</td> </tr>
  <tr> <td><a href="https://github.com/healthylaife/MIMIC-IV-Data-Pipeline">Gupta 2022</a></td> <td>MIMIC-IV</td> <td>✓</td> <td>*</td> <td>257k</td> <td>4</td> <td>--</td> <td>✓</td> <td>✓</td> <td>--</td> </tr>
  <tr> <td><a href="https://github.com/aishwarya-rm/cop-e-cat">COP-E-CAT</a></td> <td>MIMIC-IV</td> <td>✓</td> <td>*</td> <td>257k</td> <td>4</td> <td>--</td> <td>✓</td> <td>✓</td> <td>--</td> </tr>
  <tr> <td><a href="https://github.com/nliulab/mimic4ed-benchmark">Xie 2022</a></td> <td>MIMIC-IV</td> <td>✓</td> <td>*</td> <td>216k</td> <td>3</td> <td>--</td> <td>✓</td> <td>✓</td> <td>--</td> </tr>
  <tr> <td><a href="https://github.com/mostafaalishahi/eICU_Benchmark">eICU</a></td> <td>eICU</td> <td>✓</td> <td>--</td> <td>73k</td> <td>4</td> <td>--</td> <td>✓</td> <td>✓</td> <td>--</td> </tr>
  <tr> <td><a href="https://github.com/mmcdermott/comprehensive_MTL_EHR">EHR PT</a></td> <td>MIMIC-III / eICU</td> <td>✓</td> <td>--</td> <td>86k</td> <td>11</td> <td>✓</td> <td>✓</td> <td>✓</td> <td>--</td> </tr>
  <tr> <td><a href="https://github.com/MLD3/FIDDLE">FIDDLE</a></td> <td>MIMIC-III / eICU</td> <td>✓</td> <td>--</td> <td>157k</td> <td>3</td> <td>--</td> <td>✓</td> <td>✓</td> <td>--</td> </tr>
  <tr> <td><a href="https://github.com/ratschlab/HIRID-ICU-Benchmark">HiRID-ICU</a></td> <td>HiRID</td> <td>✓</td> <td>--</td> <td>33k</td> <td>6</td> <td>--</td> <td>✓</td> <td>✓</td> <td>--</td> </tr>
  <tr> <td><a href="https://www.sciencedirect.com/science/article/pii/S1532046419302564?via%3Dihub">Solares 2020</a></td> <td>CPRD</td> <td>✓</td> <td>✓</td> <td>4M</td> <td>2</td> <td>--</td> <td>--</td> <td>--</td> <td>--</td> </tr>
</table>


<a name="installation"/>

# 💿 Installation

Please use the following steps to create an environment for running the EHRSHOT benchmark.

**1)**: Install **EHRSHOT**

```bash
conda create -n EHRSHOT_ENV python=3.10 -y
conda activate EHRSHOT_ENV

git clone https://github.com/som-shahlab/ehrshot-benchmark.git
cd ehrshot-benchmark
pip install -r requirements.txt
```

**2)**: Install **FEMR**

For our data preprocessing pipeline we use **[FEMR  (Framework for Electronic Medical Records)](https://github.com/som-shahlab/femr)**, a Python package for building deep learning models with EHR data. 

You must also have CUDA/cuDNN installed (we recommend CUDA 11.8 and cuDNN 8.7.0)

Note that this currently only works on Linux machines.

```bash
pip install --upgrade "jax[cuda11_pip]==0.4.8" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
conda install bazel=6 -y
pip install git+https://github.com/som-shahlab/femr.git@ehrshot_branch

# pip install femr==0.0.20

```

## Download Private Assets

[Go to Redivis here to download](https://redivis.com/ShahLab/editor/datasets/53gc-8rhx41kgt) several assets that we cannot redistribute publicly on Github This includes the dataset and the weights of the pre-trained foundation model (CLMBR) we benchmark.

## Folder Structure

Your final folder structure should look like this:

- `ehrshot-benchmark/`
  - `EHRSHOT_ASSETS/`
    - `database/`
      - *We provide this asset from Redivis, which contains deidentified EHR data as a [FEMR](https://github.com/som-shahlab/ehrshot-femr) extract.*
    - `labels/`
      - *We provide this asset from Redivis, which contains labels and few-shot samples for all our tasks.*
    - `models/`
      - *We provide this asset from Redivis, which contains our pretrained foundation model for EHRs.*
    - `splits.csv`
      - *We provide this asset from Redivis, which determine which patient corresponds to which split.*
  - `ehrshot/`
    - *We provide the scripts to run the benchmark here*

<a name="usage"/>

# 👩‍💻 Usage

To execute the entire benchmark end-to-end, please run:

```bash
bash run_all.sh
```

# Citation

If you find this project helpful, please cite [our paper](https://arxiv.org/abs/2307.02028):

```
@article{wornow2023ehrshot,
      title={EHRSHOT: An EHR Benchmark for Few-Shot Evaluation of Foundation Models}, 
      author={Michael Wornow and Rahul Thapa and Ethan Steinberg and Jason Fries and Nigam Shah},
      year={2023},
      eprint={2307.02028},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

# License

The source code of this repo is released under the Apache License 2.0. The model license are listed on their corresponding webpages.
