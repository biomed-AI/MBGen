# MBGen: De Novo Molecular Generation from Mass Spectra via Many-Body Enhanced Diffusion

This repository contains the official implementation of **MBGen**, proposed in our AAAI-26 paper *"De Novo Molecular Generation from Mass Spectra via Many-Body Enhanced Diffusion"*.  
**MBGen** introduces a many-body enhanced diffusion framework for generating molecular structures directly from mass spectra.

---

## 🧩 Environment Installation

We recommend using **conda** for environment management.

```bash
conda create -y -c conda-forge -n mbgen rdkit=2024.09.4 python=3.9
conda activate mbgen
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu118
pip install -e .
```

---

## 📦 Dataset Download and Processing

**MBGen** supports pretraining on molecular graph data and fine-tuning on mass spectrum datasets.
Please run the scripts in the following order to download and preprocess the datasets:

```bash
bash data_processing/00_download_fp2mol_data.sh
bash data_processing/01_download_canopus_data.sh
bash data_processing/02_download_msg_data.sh
bash data_processing/03_preprocess_fp2mol.sh
```

## 🚀 Training and Evaluation

---

**🔧 Decoder Pretraining**

Run the following command to pretrain the molecular decoder:

```bash
python fp2mol_main.py
```

---

**🎯 Finetuning on Mass Spectrum Data**

Run the following command to fine-tune the model for molecule generation from MS data:

```bash
python spec2mol_main.py
```

---

