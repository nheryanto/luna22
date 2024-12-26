# 3D Transfer Learning for Lung Nodule Malignancy Binary Classification
## Steps to Run the Project

### 1. Download Data
Download the following files from [Zenodo](https://zenodo.org/records/6559584):
- `LIDC-IDRI_1176.zip`
- `LIDC-IDRI_1176.npy`

Unzip `LIDC-IDRI_1176.zip` as is. This will create a folder named `LIDC-IDRI`.

---

### 2. Create a New Environment
Create a new environment using `conda` as described in `requirements.txt`:

```bash
conda create -n luna22 python=3.10 -y
conda activate luna22
```

---

### 3. Install Required Libraries
Install the necessary libraries either manually or by running the `setup_dependencies.sh` script.

#### Manual Installation:
```bash
pip3 install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install timm-3d timm SimpleITK pandas scikit-learn torchio matplotlib torchinfo
```

---

### 4. Run the Scripts
Execute the scripts in the following order:

1. **Preprocess Labels:**
   ```bash
   python 1-preprocess-label.py
   ```

2. **Preprocess Images:**
   ```bash
   python 2-preprocess-image.py
   ```

3. **Split Nested Cross-Validation:**
   ```bash
   python 3-split-nested-cv.py
   ```

4. **Train the Model:**
   ```bash
   python 4-train.py
   ```

5. **Run Inference:**
   ```bash
   python 5-inference.py
   ```

---

### Notes
- Ensure all dependencies are installed before running the scripts.
- For GPU support, verify that your system meets the CUDA version specified in the `torch` installation.