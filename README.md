# MRI Brain Tumor Assistant

A local, offline app that looks at one brain MRI picture, **suggests the tumor type** (glioma, meningioma, pituitary tumor, or no tumor), and then **writes a short draft report for a doctor** with a small language model (Qwen2.5:3b, running in Ollama).

> **Important:** This is a student project for learning. It is **not a medical device** and it does **not** give a diagnosis. Never use it for real patient decisions.

![Grad-CAM examples](docs/gradcam_examples.png)

*Grad-CAM shows where the model looked. It is **not** a tumor outline.*

---

## What it does

1. You upload an MRI picture (or pick a test picture whose true class is known).
2. A deep learning classifier gives a probability for 4 classes.
3. The app shows a Grad-CAM heat map (where the model looked).
4. Qwen2.5:3b writes a short report (Findings, Possible meaning, Suggested next steps, Limits) using only the numbers the classifier gave. It cannot see the picture.
5. If Ollama is not running, the app uses a built-in template report instead.

## What it does NOT do

- It does **not** draw the tumor border. The dataset has only class labels, no masks, so this is classification, not segmentation.
- It does **not** measure size, location, or grade.
- It works on **one 2D picture**, not a full MRI series.

---

## Model

| | |
|---|---|
| Architecture | ResNet34 encoder + classification head, taken from `segmentation_models_pytorch` U-Net (the decoder is not used) |
| Pretrained weights | ImageNet (encoder) |
| Input | 256 x 256 RGB |
| Classes | glioma, meningioma, no tumor, pituitary |
| Loss | Cross-entropy with class weights and label smoothing 0.05 |
| Optimizer | AdamW, learning rate 3e-4, ReduceLROnPlateau (factor 0.5) |
| Augmentation | small rotation, zoom, shift, left-right flip, brightness/contrast |
| Training | 20 epochs, batch size 32, early stopping (patience 6), about 18 s per epoch on a Colab GPU |

**Dataset:** [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) by Masoud Nickparvar (also on [Zenodo](https://zenodo.org/records/12735702), CC BY 4.0). The dataset is **not** included in this repo.

**Data cleaning:** The notebook checks for duplicate pictures (correlation + pixel difference) between Training and Testing. It found and removed 0 training duplicates. The 15% validation split is taken from the Training folder. The official Testing folder is never used for training or for choosing the model.

---

## Results

Evaluated once on the official Testing folder: **1,600 pictures (400 per class)**.

| Metric | Value |
|---|---|
| Accuracy | **95.4%** |
| Macro F1 | **0.953** |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Glioma | 1.000 | **0.830** | 0.907 |
| Meningioma | 0.898 | 0.990 | 0.942 |
| No tumor | 0.946 | 1.000 | 0.972 |
| Pituitary | 0.988 | 0.998 | 0.993 |

![Confusion matrix](docs/confusion_matrix.png)

### Honest notes about these numbers

- **Glioma is the weak class.** Out of 400 real gliomas, the model called 44 "meningioma" and 23 "no tumor". The "no tumor" mistakes are the most serious ones in a medical setting.
- **Validation F1 (0.993) is higher than test F1 (0.953).** The validation pictures come from the Training folder, so they are easier. The test number is the honest one.
- **The dataset has no patient ID.** Pictures from the same person may appear in train, validation, and test, so real-life scores can be lower.
- The model was trained on one public dataset only. It was not tested on MRI from other hospitals or scanners.

![Training curves](docs/training_curves.png)

---

## Project structure

```
.
├── app_colab.py                 # Streamlit app (run this)
├── braintumor.py                # Classifier class, Grad-CAM, command-line predict
├── medical_assistant.py         # Ollama / Qwen helpers (start, settings, disclaimer)
│                                #   also holds an earlier LGG segmentation pipeline,
│                                #   which the current app does not use
├── braintumor_colab.ipynb       # Colab notebook: download data, train, test, Grad-CAM
├── models/                      # put braintumor_unet_resnet34_cls.pt here (not in repo)
├── outputs/braintumor/          # test_results.json, training_log.csv
└── docs/                        # images used in this README
```

---

## How to run

**1. Install Python libraries** (Python 3.10+; install PyTorch first from [pytorch.org](https://pytorch.org) for your GPU or CPU):

```bash
pip install -r requirements.txt
```

**2. Install Ollama and the language model**

```bash
# install Ollama from https://ollama.com, then:
ollama pull qwen2.5:3b
```

The app starts Ollama and loads Qwen by itself when it runs.

**3. Get the trained model**

Download `braintumor_unet_resnet34_cls.pt` from the [Releases page](../../releases) and put it in the `models/` folder. Or train it yourself (next section).

**4. Start the app**

```bash
streamlit run app_colab.py
```

**Optional: test the classifier alone**

```bash
python braintumor.py predict "path/to/picture.jpg"
```

The first time you use "Test picture", the app can download the dataset's Testing folder (about 150 MB, needs internet).

## How to train the model yourself

1. Open `braintumor_colab.ipynb` in [Google Colab](https://colab.research.google.com).
2. Set the runtime to **GPU** (Runtime -> Change runtime type).
3. Run all cells. The notebook downloads the dataset, trains, tests, and makes a zip file.
4. Copy `braintumor_unet_resnet34_cls.pt` into `models/`, and the result files into `outputs/braintumor/`.

---

## Safety design for the report

The language model gets strict rules: use only the given numbers, call the result an "AI suggestion" (never a diagnosis), say clearly when the model is unsure (top probability below 60%) or when the picture looks different from the training data, say that "no tumor" does not rule out disease, give no treatment or drug advice, and remind the reader that a radiologist must review the original images. Every report ends with a disclaimer.

## Limitations

- Only 2D classification of single pictures; no tumor outline, size, or grade.
- Glioma recall is 83% on the test set.
- Trained on one public dataset without patient IDs; real-world performance is unknown.
- A 3B language model can still make mistakes in wording. Its output is a draft, not a medical text.

## Future work

- Test on data from other sources and report the difference.
- Add real tumor segmentation using a dataset that has masks.
- Improve glioma recall (more data, better augmentation, threshold tuning).
- Evaluate the report quality with a doctor or radiology student.

## Credits

- Dataset: Masoud Nickparvar, *Brain Tumor MRI Dataset* (CC BY 4.0).
- Models and libraries: PyTorch, `segmentation_models_pytorch`, Streamlit, Ollama, Qwen2.5 (downloaded separately; see its own license).

## Author

**Jassim** — Artificial Intelligence Engineering student, Near East University.
LinkedIn: [add your link here] · GitHub: [add your link here]

## License

MIT (see `LICENSE`).
