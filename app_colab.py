
import os
import random
import time
import json
import csv
from pathlib import Path

import numpy as np
import requests
import streamlit as st
from PIL import Image

import medical_assistant as ma      # only used for: project folder, Ollama/Qwen settings and start-up
import braintumor as bt             # the Colab classifier

st.set_page_config(page_title="MRI Assistant", layout="wide")

DATASET = "masoudnickparvar/brain-tumor-mri-dataset"
TIME_LIMIT = 10.0                                   # the professor's limit: analysis + report in under 10 s
PICTURE_TYPES = ["jpg", "jpeg", "png", "tif", "tiff", "bmp"]


# =====================================================================================
# LOADING (done one time, then kept in memory)
# =====================================================================================
@st.cache_resource(show_spinner="Loading the Colab model...")
def load_classifier():
    try:
        cls = bt.TumorClassifier()
        cls.analyze(Image.new("RGB", (cls.meta["image_size"],) * 2))      # warm-up, so the first real picture is fast
        return cls, None
    except Exception as e:
        return None, str(e)


@st.cache_resource
def start_qwen():
    ma.start_llm_in_background()        # starts Ollama and loads Qwen (does nothing if it is already running)
    return True


def pct(p):
    return round(100 * float(p), 1)


# =====================================================================================
# INPUT CHECK: is this picture like the pictures the model learned?
# =====================================================================================
def input_warnings(image, name):
    warnings = []
    rgb = np.array(image.convert("RGB"), dtype=np.int16)
    h, w = rgb.shape[:2]
    # the Colab pictures are grey (all 3 colors the same). LGG .tif slices have 3 different MRI sequences.
    color_diff = (np.abs(rgb[..., 0] - rgb[..., 1]).mean() + np.abs(rgb[..., 1] - rgb[..., 2]).mean()) / 2
    if color_diff > 6:
        warnings.append("the picture has different color channels (like the LGG .tif slices); the model learned grey MRI pictures, so this result may be wrong")
    elif name.lower().endswith((".tif", ".tiff")):
        warnings.append("this is a .tif file; the model learned .jpg pictures from the Brain Tumor MRI dataset, so check the result carefully")
    if min(h, w) < 100:
        warnings.append(f"the picture is very small ({w}x{h})")
    if max(h, w) > 1.5 * min(h, w):
        warnings.append(f"the picture is not square ({w}x{h}); it may be a screenshot, not one MRI picture")
    return warnings


# =====================================================================================
# QWEN REPORT (its own rules, made for the classifier; the backend file is not changed)
# =====================================================================================
QWEN_RULES = """You are a clinical report assistant that helps a doctor read the output of an AI image classifier for brain MRI.
You are talking to the doctor. Be professional, short, and clear. Use simple English.

WHAT THE AI MODEL DID
- A classifier looked at ONE brain MRI picture and gave a probability for 4 classes: glioma, meningioma, pituitary tumor, no tumor.
- It was trained on the public Brain Tumor MRI dataset. It does NOT measure the tumor, does NOT mark its border or location, and does NOT know the grade or other diseases.
- You CANNOT see the picture. You only get the facts below.

STRICT RULES
1. Use ONLY the given facts. Never invent sizes, location, grade, stage, survival, or a final diagnosis.
2. Call the tumor type an "AI suggestion", not a diagnosis. Use only the given probabilities.
3. If the model is unsure, or there are warnings, say clearly that the result is uncertain.
4. If the top class is "No tumor", say that this does NOT rule out disease.
5. Do not give treatment orders or drug doses.
6. If the doctor asks a question, answer it briefly inside these limits. If the facts cannot answer it, say that.
7. Say that the final decision belongs to the doctor and that a radiologist should review the original images.

OUTPUT FORMAT (plain text, no markdown symbols, max 150 words)
Findings: 1-2 sentences with the top class and its probability.
Possible meaning: 1-2 careful sentences ("may suggest", "is consistent with").
Suggested next steps: 2-3 short items.
Limits: 1 sentence.
"""


def facts_text(res, warnings):
    lines = [f"- top class: {res['display']} ({pct(res['confidence'])}% probability)",
             "- all probabilities: " + ", ".join(f"{bt.DISPLAY.get(k, k)} {pct(v)}%" for k, v in res["probs"].items()),
             f"- model is unsure (top probability below {int(100 * bt.LOW_CONFIDENCE)}%): {'yes' if res['uncertain'] else 'no'}",
             "- warnings: " + ("; ".join(warnings) if warnings else "none")]
    return "\n".join(lines)


def template_report(res, warnings):
    """used when Qwen is off or not working."""
    name, p = res["display"], pct(res["confidence"])
    if res["label"] == "notumor":
        text = (f"Findings: the AI classifier suggests no tumor in this picture ({p}% probability).\n"
                "Possible meaning: no tumor pattern was recognized in this single picture. This does not rule out disease.\n")
    else:
        text = (f"Findings: the AI classifier suggests {name.lower()} ({p}% probability).\n"
                f"Possible meaning: the picture may show features consistent with {name.lower()}. This is an AI suggestion, not a diagnosis.\n")
    text += "Suggested next steps: review the full MRI series, compare other sequences, and correlate clinically.\n"
    if res["uncertain"]:
        text += f"Warning: the model is not sure (top probability {p}%).\n"
    if warnings:
        text += "Warnings: " + "; ".join(warnings) + "\n"
    text += "Limits: one picture only, one AI model, it cannot measure the tumor or see other diseases."
    return text


def write_report(res, warnings, notes, question, use_qwen):
    start = time.perf_counter()
    error = None
    if use_qwen:
        user = "AI classifier results for one MRI picture:\n" + facts_text(res, warnings)
        if notes.strip():
            user += "\n\nDoctor's clinical notes:\n" + notes.strip()
        if question.strip():
            user += "\n\nDoctor's question:\n" + question.strip()
        user += "\n\nWrite the report now."
        try:
            r = requests.post(f"{ma.OLLAMA_URL}/api/chat", timeout=ma.OLLAMA_TIMEOUT, json={
                "model": ma.OLLAMA_MODEL, "stream": False, "keep_alive": ma.OLLAMA_KEEP_ALIVE,
                "messages": [{"role": "system", "content": QWEN_RULES}, {"role": "user", "content": user}],
                "options": {"temperature": ma.OLLAMA_TEMPERATURE, "num_predict": ma.OLLAMA_MAX_TOKENS,
                            "num_ctx": ma.OLLAMA_CONTEXT}})
            r.raise_for_status()
            text = r.json()["message"]["content"].strip()
            if not text:
                error = "Qwen returned an empty answer"
        except requests.exceptions.ConnectionError:
            error = "Ollama is not running"
        except requests.exceptions.Timeout:
            error = f"Qwen took longer than {ma.OLLAMA_TIMEOUT} seconds"
        except Exception as e:
            error = f"AI assistant error: {e}"
    if not use_qwen or error:
        text = template_report(res, warnings)
        source = "built-in template (AI assistant switched off)" if not use_qwen else "built-in template (Qwen not used)"
    else:
        source = ma.OLLAMA_MODEL
    return {"text": text + "\n\n" + ma.DISCLAIMER, "source": source, "error": error,
            "seconds": round(time.perf_counter() - start, 2)}


# =====================================================================================
# TEST PICTURES (the Testing folder of the Colab dataset: the folder name = the true answer)
# =====================================================================================
def class_key(folder_name):
    key = folder_name.lower().replace("_", "").replace("-", "").replace(" ", "")
    return {"notumour": "notumor"}.get(key, key)


@st.cache_data(show_spinner=False)
def class_folders(folder_text):
    """{class: [picture paths]} for a folder that has glioma / meningioma / notumor / pituitary inside."""
    folder, out = Path(folder_text), {}
    if not folder.is_dir():
        return out
    for sub in sorted(folder.iterdir()):
        if sub.is_dir() and class_key(sub.name) in bt.DISPLAY:
            files = sorted(str(p) for p in sub.iterdir() if p.suffix.lower().lstrip(".") in PICTURE_TYPES)
            if files:
                out[class_key(sub.name)] = files
    return out


@st.cache_data(show_spinner="Looking for the test pictures...")
def find_test_folder(_refresh=0):
    roots = [ma.PROJECT_DIR / "data" / "raw" / "datasets" / "masoudnickparvar",
             Path.home() / ".cache" / "kagglehub" / "datasets" / "masoudnickparvar",
             ma.PROJECT_DIR / "data" / "braintumor"]
    for root in roots:
        if root.exists():
            for d in sorted(root.rglob("*")):
                if d.is_dir() and d.name.lower() == "testing" and len(class_folders(str(d))) >= 2:
                    return str(d)
    return None


def test_folder():
    typed = st.session_state.get("own_folder", "").strip().strip('"')
    return typed or find_test_folder(st.session_state.get("refresh", 0))


def download_dataset():
    os.environ["KAGGLEHUB_CACHE"] = str(ma.PROJECT_DIR / "data" / "raw")      # save inside the project folder
    import kagglehub
    return kagglehub.dataset_download(DATASET)


def no_test_pictures_box():
    st.info("The test pictures were not found. They are the 'Testing' folder of the Colab dataset "
            "(1,600 pictures, the folder name is the true answer). Download them one time (about 150 MB, needs internet), "
            "or type the folder in the sidebar.")
    if st.button("Download the test pictures"):
        with st.spinner("Downloading... this can take a few minutes"):
            try:
                path = download_dataset()
                st.success(f"Downloaded to: {path}")
            except Exception as e:
                st.error(f"Download failed: {e}")
        st.session_state.refresh = st.session_state.get("refresh", 0) + 1
        find_test_folder.clear()
        st.rerun()


# =====================================================================================
# SIDEBAR
# =====================================================================================
@st.fragment(run_every=3)
def qwen_status():
    state, msg = ma.LLM_STATE["state"], ma.LLM_STATE["message"]
    if state == "ready":
        st.success(f"Qwen (AI assitant writer): ready  \n{msg}")
    elif state == "error":
        st.error(f"AI assistant: not available  \n{msg}")
    else:
        st.info(f"Qwen: {msg}")


def sidebar(cls, error):
    st.sidebar.title("MRI Tumor Type")
    page = st.sidebar.radio("Page", ["Analyze", "Colab results"])
    st.sidebar.divider()
    if cls:
        st.sidebar.success("Model: ready")
    else:
        st.sidebar.error(f"Model: not loaded  \n{error}")
    with st.sidebar:
        qwen_status()
    use_qwen = st.sidebar.toggle("Write the report with the AI assistant", value=True)
    st.sidebar.divider()
    st.sidebar.text_input("Test pictures folder (optional)", key="own_folder",
                          help="Leave empty to find the dataset's Testing folder by itself. "
                               "Or paste a folder that has glioma / meningioma / notumor / pituitary folders inside.")
    folder = test_folder()
    if folder and class_folders(folder):
        n = sum(len(v) for v in class_folders(folder).values())
        st.sidebar.caption(f"Test pictures: {n} found in  \n{folder}")
    else:
        st.sidebar.caption("Test pictures: not found")
    return page, use_qwen


# =====================================================================================
# PAGE 1: ANALYZE ONE PICTURE
# =====================================================================================
def run_analysis(cls, image, name, truth, notes, question, use_qwen):
    start = time.perf_counter()
    res = cls.analyze(image)
    warnings = input_warnings(image, name)
    report = write_report(res, warnings, notes, question, use_qwen)
    return {"id": (name, truth), "name": name, "truth": truth, "res": res, "warnings": warnings,
            "report": report, "picture": image.convert("RGB"),
            "total": round(time.perf_counter() - start, 2)}


def page_analyze(cls, use_qwen):
    st.title("Analyze an MRI picture")
    st.caption("The Colab model names the tumor type, then AI assistant writes a short report. "
               "It does not draw the tumor border (the Colab dataset has no masks).")
    source = st.radio("Picture", ["Upload a picture", "Test picture (true answer known)"], horizontal=True)
    image = name = truth = None

    if source == "Upload a picture":
        up = st.file_uploader("Drop one MRI picture here", type=PICTURE_TYPES)
        if up is not None:
            try:
                image = Image.open(up)
                image.load()
                name = up.name
            except Exception as e:
                st.error(f"This file could not be opened as a picture: {e}")
    else:
        folder = test_folder()
        groups = class_folders(folder) if folder else {}
        if not groups:
            no_test_pictures_box()
        else:
            c1, c2 = st.columns([2, 1])
            truth = c1.selectbox("True class (the folder name)", list(groups), format_func=lambda k: bt.DISPLAY.get(k, k))
            files = groups[truth]
            key = f"pick_{truth}"
            c2.write("")
            if c2.button("Random picture", width="stretch"):
                st.session_state[key] = random.choice(files)
            path = st.selectbox(f"Picture ({len(files)} in this class)", files, key=key, format_func=lambda p: Path(p).name)
            image = Image.open(path)
            image.load()
            name = Path(path).name

    notes = st.text_area("Doctor's notes (optional)", height=80)
    question = st.text_input("Question for the AI assistant(optional)")

    if image is not None and st.button("Analyze", type="primary"):
        if cls is None:
            st.error("The model is not loaded. See the sidebar.")
        else:
            with st.spinner("The model is looking, then the AI assistant is writing..."):
                st.session_state.result = run_analysis(cls, image, name, truth, notes, question, use_qwen)

    result = st.session_state.get("result")
    if image is not None and result and result["id"] == (name, truth):
        show_result(result)
    elif image is not None:
        st.image(image.convert("RGB"), caption=name, width=320)


def show_result(r):
    res, rep = r["res"], r["report"]
    left, right = st.columns([1, 1])
    with left:
        view = st.radio("Show", ["Picture", "Grad-CAM (where the model looked)"], horizontal=True)
        if view == "Picture":
            st.image(r["picture"], caption=r["name"], width="stretch")
        else:
            alpha = st.slider("Heat map strength", 0.0, 1.0, 0.45, 0.05)
            st.image(bt.cam_overlay(res["rgb"], res["cam"], alpha), width="stretch",
                     caption="red = the model looked here, blue = ignored. This is NOT a tumor outline.")
    with right:
        st.subheader(f"AI suggestion: {res['display']}")
        st.write(f"probability **{pct(res['confidence'])}%**")
        if r["truth"]:
            true_name = bt.DISPLAY.get(r["truth"], r["truth"])
            if r["truth"] == res["label"]:
                st.success(f"Correct. True class: {true_name}")
            else:
                st.error(f"Wrong. True class: {true_name}")
        if res["uncertain"]:
            st.warning(f"The model is not sure (top probability under {int(100 * bt.LOW_CONFIDENCE)}%).")
        for w in r["warnings"]:
            st.warning(w[0].upper() + w[1:])
        for label, p in sorted(res["probs"].items(), key=lambda kv: -kv[1]):
            st.progress(float(p), text=f"{bt.DISPLAY.get(label, label)}  {pct(p)}%")
        c1, c2, c3 = st.columns(3)
        c1.metric("Model time", f"{res['seconds']:.2f} s")
        c2.metric("Report time", f"{rep['seconds']:.2f} s")
        c3.metric("Total", f"{r['total']:.2f} s")
       
    st.subheader("Report")
    st.caption(f"Written by: {rep['source']}" + (f"  ({rep['error']})" if rep["error"] else ""))
    with st.container(border=True):
        st.text(rep["text"])


# =====================================================================================
# THE COLAB RESULTS FILE
# =====================================================================================
def colab_results():
    f = bt.paths().out / "test_results.json"
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


# =====================================================================================
# PAGE 2: THE COLAB RESULTS (from the zip)
# =====================================================================================
def page_colab(cls):
    st.title("Colab training results")
    out = bt.paths().out
    colab = colab_results()
    if not colab:
        st.warning(f"test_results.json was not found in {out}. Copy the files from braintumor_results.zip into that folder.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test accuracy", f"{100 * colab['accuracy']:.1f}%")
        c2.metric("Test macro F1", f"{colab['macro_f1']:.3f}")
        c3.metric("Test pictures", colab["n_test"])
        c4.metric("Best epoch", colab.get("checkpoint_epoch", "?"))
        rows = [{"class": bt.DISPLAY.get(c, c), "pictures": int(v["support"]), "precision": round(v["precision"], 3),
                 "recall": round(v["recall"], 3), "F1": round(v["f1-score"], 3)} for c, v in colab["per_class"].items()]
        st.dataframe(rows, hide_index=True, width="stretch")
        weakest = min(rows, key=lambda r: r["recall"])
        st.info(f"Weakest class: {weakest['class']} (recall {weakest['recall']:.2f}). "
                "The model misses some of these pictures and calls them another class, so a negative answer is never proof.")

    log = out / "training_log.csv"
    if log.exists():
        with open(log, newline="", encoding="utf-8") as f:
            history = list(csv.DictReader(f))
        if history:
            best_val = max(float(h["val_f1"]) for h in history)
            st.caption(f"Training: {len(history)} epochs, about {np.mean([float(h['seconds']) for h in history]):.0f} s per epoch on Colab. "
                       f"Best validation F1 {best_val:.3f}.")
            if colab and best_val - colab["macro_f1"] > 0.02:
                st.warning(f"Validation F1 ({best_val:.3f}) is higher than test F1 ({colab['macro_f1']:.3f}). "
                           "The validation pictures came from the Training folder and were easier. The test number is the honest one.")
    if colab and colab.get("note"):
        st.caption("Note from Colab: " + colab["note"])

    for file, title in [("confusion_matrix.png", "Confusion matrix (test)"), ("training_curves.png", "Training curves"),
                        ("gradcam_examples.png", "Grad-CAM examples"), ("examples.png", "Example pictures")]:
        if (out / file).exists():
            st.subheader(title)
            st.image(str(out / file), width="stretch")


# =====================================================================================
def main():
    cls, error = load_classifier()
    start_qwen()
    page, use_qwen = sidebar(cls, error)
    if page == "Analyze":
        page_analyze(cls, use_qwen)
    else:
        page_colab(cls)


main()