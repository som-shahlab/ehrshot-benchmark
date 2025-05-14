# 👂 💉 EHRSHOT

A benchmark/dataset for few-shot evaluation of foundation models for electronic health records (EHRs). You can **[read the paper here](https://arxiv.org/abs/2307.02028)**. 

Whereas most prior EHR benchmarks are limited to the ICU setting, **EHRSHOT** contains the **full longitudinal health records of 6,739 patients from Stanford Medicine** and a diverse set of **15 classification tasks** tailored towards few-shot evaluation of pre-trained models. 

> [!NOTE]  
> EHRSHOT can now be downloaded as a [MEDS compatible dataset](https://github.com/Medical-Event-Data-Standard). Please visit this [**Redivis link**](https://redivis.com/datasets/53gc-8rhx41kgt) and download the file called `EHRSHOT_MEDS.zip`

# 📖 Table of Contents
1. [Quick Start](#quick_start)
2. [Pre-trained Foundation Model](#models)
3. [Dataset + Tasks](#dataset)
4. [Comparison to Prior Work](#prior_work)
5. [Other](#other)
6. [Citation](#citation)

<a name="quick_start"/>

# 🚀 Quick Start

Use the following steps to run the EHRSHOT benchmark.

**1)**: Install **EHRSHOT**

```bash
conda create -n EHRSHOT_ENV python=3.10 -y
conda activate EHRSHOT_ENV

git clone https://github.com/som-shahlab/ehrshot-benchmark.git
cd ehrshot-benchmark
pip install -r requirements.txt --no-cache-dir
```

**2)**: Install **FEMR**

For our data preprocessing pipeline we use **[FEMR  (Framework for Electronic Medical Records)](https://github.com/som-shahlab/femr)**, a Python package for building deep learning models with EHR data. 

You must also have CUDA/cuDNN installed (we recommend CUDA 11.8 and cuDNN 8.7.0). 

> [!CAUTION]
> This currently only works on Linux machines.

```bash
pip install femr-cuda==0.1.16 dm-haiku==0.0.9 optax==0.1.4
pip install --upgrade "jax[cuda]==0.4.8" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

**3)**: **Download dataset + model** from [Redivis here](https://redivis.com/datasets/53gc-8rhx41kgt) and place the results in a directory called `EHRSHOT_ASSETS/`.

**4)**: **Run** the benchmark end-to-end with:

```bash
bash run_all.sh
```

## Folder Structure

Your final folder structure should look like this:

- `ehrshot-benchmark/`
  - `EHRSHOT_ASSETS/`
    - `athena_download/`
      - *We do NOT provide this asset. You will have to follow the instructions in the section "Downloading the Athena Ontology" below. However, you can skip this entirely by using the FEMR extract included in our Redivis download.*
    - `benchmark/`
      - *We provide this asset from Redivis, which contains labels + few-shot samples for all our tasks.*
    - `data/`
      - *We provide this asset from Redivis, which contains a CSV containing the entire dataset.*
    - `features/`
      - *We provide this asset from Redivis, which contains preprocessed count + CLMBR-based featurizations.*
    - `femr/`
      - *We provide this asset from Redivis, which contains deidentified EHR data as a [FEMR](https://github.com/som-shahlab/ehrshot-femr) extract.*
    - `figures/`
      - *We provide this asset from Redivis, which contains figures summarizing the expected results of running our benchmark.*
    - `models/`
      - *We provide this asset from Redivis, which contains the weights of our pretrained foundation model (CLMBR).*
    - `results/`
      - *We provide this asset from Redivis, which contains raw results from our running of our benchmark on the baseline models.*
    - `splits/`
      - *We provide this asset from Redivis, which determine which patient corresponds to which split.*
  - `ehrshot/`
    - *We provide the scripts to run the benchmark here*

<a name="models"/>

# 🔮 Foundation Model for EHRs

**Access:** [The model is on HuggingFace here](https://huggingface.co/StanfordShahLab/clmbr-t-base) and requires signing a research usage agreement.

We publish the model weights of a **141 million parameter** clinical foundation model pre-trained on the deidentified structured EHR data of **2.57M patients** from Stanford Medicine.

We are [one of the first](https://arxiv.org/abs/2303.12961) to fully release such a model for coded EHR data; in contrast, most prior models released for clinical data  (e.g. GatorTron, ClinicalBERT) only work with unstructured text and cannot process the rich, structured data within an EHR.

We use [Clinical Language-Model-Based Representations (CLMBR)](https://www.sciencedirect.com/science/article/pii/S1532046420302653) as our model. CLMBR is an autoregressive model designed to predict the next medical code in a patient's timeline given previous codes. CLMBR employs causally masked local attention, ensuring forward-only flow of information which is vital for prediction tasks and is in contrast to BERT-based models which are bidirectional in nature. We utilize a transformer as our base model with 141 million trainable parameters and a next code prediction objective, providing minute-level EHR resolution rather than the day-level aggregation of the original model formulation. 


<a name="dataset"/>

# 🗃️ Dataset + Tasks

**Access:** We provide two versions of EHRSHOT (each version contains identical data, just different formats). Access requires signing a research usage agreement.

1. **Original:** [Link](https://redivis.com/datasets/53gc-8rhx41kgt). The original EHRSHOT dataset from the paper (and version compatible with this repo) is stored in `EHRSHOT_ASSETS.zip` at this link.
2. **MEDS:** [Link](https://redivis.com/datasets/53gc-8rhx41kgt). The EHRSHOT dataset in a [MEDS](https://github.com/Medical-Event-Data-Standard) compatible version is stored in `EHRSHOT_MEDS.zip` at this link.

EHRSHOT contains:
* **6,739 patients**
* **41.6 million clinical events**
* **921,499 visits**
* **15 prediction tasks**

Each patient consists of an ordered timeline of clinical events taken from the structured data of their EHR (e.g. diagnoses, procedures, prescriptions, etc.). 

Each task is a predictive classification task, and includes a canonical train/val/test split. The tasks are defined as follows:

|         Task         | Type              | Prediction Time                       | Time Horizon           | Possible Label Values in Dataset |
|:--------------------:|-------------------|---------------------------------------|------------------------|--------|
| Long Length of Stay  | Binary            | 11:59pm on day of admission           | Admission duration     |  {0,1} aka {<7 days, >=7 days} |
| 30-day Readmission   | Binary            | 11:59pm on day of discharge           | 30-days post discharge |  {0,1} aka {no readmission, readmission} |
| ICU Transfer         | Binary            | 11:59pm on day of admission           | Admission duration     |  {0,1} aka {no transfer, transfer} |
| Thrombocytopenia     | 4-way Multiclass  | Immediately before result is recorded | Next result            |  {0,1,2,3} aka {low, medium, high, abnormal}  |
| Hyperkalemia         | 4-way Multiclass  | Immediately before result is recorded | Next result            |  {0,1,2,3} aka {low, medium, high, abnormal}  |
| Hypoglycemia         | 4-way Multiclass  | Immediately before result is recorded | Next result            |  {0,1,2,3} aka {low, medium, high, abnormal}  |
| Hyponatremia         | 4-way Multiclass  | Immediately before result is recorded | Next result            |  {0,1,2,3} aka {low, medium, high, abnormal}  |
| Anemia               | 4-way Multiclass  | Immediately before result is recorded | Next result            |  {0,1,2,3} aka {low, medium, high, abnormal}  |
| Hypertension         | Binary            | 11:59pm on day of discharge           | 1 year post-discharge  |  {0,1} aka {no diagnosis, diagnosis} |
| Hyperlipidemia       | Binary            | 11:59pm on day of discharge           | 1 year post-discharge  |  {0,1} aka {no diagnosis, diagnosis} |
| Pancreatic Cancer    | Binary            | 11:59pm on day of discharge           | 1 year post-discharge  |  {0,1} aka {no diagnosis, diagnosis} |
| Celiac               | Binary            | 11:59pm on day of discharge           | 1 year post-discharge  |  {0,1} aka {no diagnosis, diagnosis} |
| Lupus                | Binary            | 11:59pm on day of discharge           | 1 year post-discharge  |  {0,1} aka {no diagnosis, diagnosis} |
| Acute MI             | Binary            | 11:59pm on day of discharge           | 1 year post-discharge  |  {0,1} aka {no diagnosis, diagnosis} |
| Chest X-Ray Findings | 14-way Multilabel | 24hrs before report is recorded       | Next report            |  {0,1,...,8192} aka binary string where a 1 at location `idx` means that the label at `CHEXPERT_LABELS[idx]` is True, per [this array](https://github.com/som-shahlab/ehrshot-benchmark/blob/f23b83a2b487b6ae8da06cb08b23b3656c447307/ehrshot/utils.py#L107C1-L122C2) |



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

<a name="other"/>

# Other

## Downloading the Athena Ontology

The FEMR extract provided in the Redivis download contains all the necessary concepts, so you can ignore this so long as you skip running the bash script `1_create_femr_database.sh`.

If you want to recreate the FEMR extract from scratch, however, then you'll need to download the Athena ontology yourself:
1. Go to the [Athena website at this link](https://athena.ohdsi.org/vocabulary/list). You may need to create an account.
2. Click the green "Download" button at the top right of the website
3. Click the purple "Download Vocabularies" button below the green "Download" button
4. Name the bundle "athena_download" and select 5.x version
5. Scroll to the bottom of the list, and click the blue "Download" button
6. It will take some time for the download to be ready. Please [refresh the webpage here](https://athena.ohdsi.org/vocabulary/download-history) to check whether your download is ready. Once the download is ready, click "Download"
7. After the download is complete, unzip the file and move all the files into the `EHRSHOT_ASSETS/athena_download/` folder in your repo.

After downloading the Athena OHDSI Ontology, you will have to separately download the CPT subset of the ontology. You can follow the instructions in the `readme.txt` in your Athena download, or follow the steps below:

1. Create a [UMLS account here](https://uts.nlm.nih.gov/uts/signup-login)
2. Get your [UMLS API key here](https://uts.nlm.nih.gov/uts/edit-profile)
3. From the `EHRSHOT_ASSETS/athena_download/` folder, run this command: `bash cpt.sh <YOUR UMLS API KEY>`

Your ontology will then be ready to go!

## Scripts

We've included a number of helper scripts in `ehrshot/scripts`. None of these are needed to run the EHRSHOT benchmark, but they provide useful functionality as follows:

* [confirm_labeled_patients_are_equal](ehrshot/scripts/confirm_labeled_patients_are_equal.py) - confirms that two `EHRSHOT_ASSETS/benchmark/` folders contain identical labels in their `labeled_patients.csv` files
* [convert_clmbr_dictionary_to_json](ehrshot/scripts/convert_clmbr_dictionary_to_json.py) - converts the [CLMBR vocabulary file](/EHRSHOT_ASSETS/models/clmbr/dictionary) into a human readable `.json` file.
* [convert_to_meds](ehrshot/scripts/convert_to_meds.py) - converts EHRSHOT into the [MEDS dataset format](https://github.com/Medical-Event-Data-Standard/meds/tree/main)
* [create_splits](ehrshot/scripts/create_splits.py) - creates splits for other FEMR extracts into the EHRSHOT format
* [generate_benchmark_starr](ehrshot/scripts/generate_benchmark_starr.py) - runs EHRSHOT labelers on entire Stanford STARR-OMOP dataset

<a name="citation"/>

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

The source code of this repo is released under the Apache License 2.0. The model license and dataset license are listed on their corresponding webpages.
