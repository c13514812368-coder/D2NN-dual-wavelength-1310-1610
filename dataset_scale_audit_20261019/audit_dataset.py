"""Read-only dataset diagnosis: can coarse visual information predict fine labels?

These fixed SVM probes diagnose available cues; they are not optical experiments.
No hyperparameter search, no use of validation labels for fitting.
"""
from pathlib import Path
import json
import time

import numpy as np
from PIL import Image, ImageOps, ImageFilter
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
ROOT = PROJECT / "data/imagenet_1k_sd_scene_3coarse_fixed"
COARSE = ["natural_landform", "water_coast", "built_environment"]
FINE = ["n09193705_alp", "n09468604_valley", "n09472597_volcano",
        "n09256479_coral_reef", "n09332890_lakeside", "n09428293_seashore",
        "n03028079_church", "n02980441_castle", "n03661043_library"]


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def records():
    d = json.loads((PROJECT / "feature_upgrade_20261011/diagnostics.json").read_text(encoding="utf-8"))
    result = {}
    for split in ("train", "val"):
        paths, labels = [], []
        for old in d["path_manifest"][split]:
            relative = old.split("/"+split+"/", 1)[1]
            path = ROOT / split / relative
            assert path.is_file()
            paths.append(path)
            labels.append(FINE.index(relative.split("/")[1]))
        result[split] = paths, np.asarray(labels)
    return result


def features(paths):
    result = {key: [] for key in ("blur4_gray4", "blur4_rgb4", "blur4_gray8")}
    for path in paths:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB").resize((64,64), Image.Resampling.BILINEAR)
            # Fixed Gaussian smoothing on the same 64x64 grid as the original pipeline.
            blurred = image.filter(ImageFilter.GaussianBlur(radius=4.))
            for key, mode, size in (("blur4_gray4", "L", 4), ("blur4_rgb4", "RGB", 4), ("blur4_gray8", "L", 8)):
                arr = np.asarray(blurred.convert(mode).resize((size,size), Image.Resampling.BILINEAR), dtype=np.float32)/255.
                result[key].append(arr.ravel())
    return {key: np.asarray(value) for key,value in result.items()}


def probe(a, b, yt, yv):
    model = make_pipeline(StandardScaler(), SVC(C=8., gamma="scale", kernel="rbf"))
    model.fit(a, yt)
    pred = model.predict(b)
    return dict(val_accuracy=float((pred==yv).mean()), confusion_matrix=confusion_matrix(yv,pred).tolist())


def contact_sheet(rec):
    paths, labels = rec["val"]
    rng = np.random.default_rng(20261019)
    fig, axes = plt.subplots(9, 6, figsize=(12,16))
    selected = []
    for c, name in enumerate(FINE):
        ids = rng.choice(np.flatnonzero(labels==c), 3, replace=False)
        for j, idx in enumerate(ids):
            path = paths[idx]
            selected.append(str(path.relative_to(ROOT)))
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB").resize((64,64),Image.Resampling.BILINEAR)
                blur = image.filter(ImageFilter.GaussianBlur(radius=4)).convert("L").resize((8,8),Image.Resampling.BILINEAR)
                axes[c,j*2].imshow(image)
                axes[c,j*2+1].imshow(blur,cmap="gray",vmin=0,vmax=255,interpolation="nearest")
            if c == 0:
                axes[c,j*2].set_title("Original 64x64",fontsize=10)
                axes[c,j*2+1].set_title("Blur + 8x8 gray",fontsize=10)
        for ax in axes[c]:
            ax.set_xticks([])
            ax.set_yticks([])
        axes[c,0].set_ylabel(name.split("_",1)[1],rotation=0,ha="right",va="center",fontsize=10)
    fig.suptitle("Existing validation data: original vs deliberately coarse representation",fontsize=13)
    fig.subplots_adjust(left=.13,right=.99,bottom=.02,top=.95,wspace=.06,hspace=.1)
    fig.savefig(HERE/"original_vs_coarse.png",dpi=160)
    plt.close(fig)
    save_json(HERE/"contact_sheet_samples.json",selected)


def main():
    HERE.mkdir(exist_ok=True)
    started = time.time()
    rec = records()
    contact_sheet(rec)
    cache = HERE/"coarse_features.npz"
    if not cache.exists():
        arrays = {}
        for split in ("train","val"):
            arrays.update({split+"_"+key:value for key,value in features(rec[split][0]).items()})
            print("Extracted "+split,flush=True)
        np.savez_compressed(cache,**arrays)
    data = np.load(cache)
    yt,yv = rec["train"][1],rec["val"][1]
    results = dict(description="Fixed electronic SVM probes on blurred low-resolution features; not wavelength or optical results.",
                   fit="train only", validation="existing repeatedly inspected val; exploratory, not independent test",
                   blur="PIL GaussianBlur radius4 on 64x64 image, followed by bilinear resize", samples=dict(train=len(yt),val=len(yv)),
                   chance=dict(coarse3=1/3,fine9=1/9,within_true_coarse=1/3), representations={})
    for key in ("blur4_gray4","blur4_rgb4","blur4_gray8"):
        a,b = data["train_"+key],data["val_"+key]
        row = dict(dimensions=a.shape[1], coarse3=probe(a,b,yt//3,yv//3),fine9=probe(a,b,yt,yv),within_true_coarse={})
        for c,name in enumerate(COARSE):
            tr,va = yt//3==c,yv//3==c
            row["within_true_coarse"][name]=probe(a[tr],b[va],yt[tr]%3,yv[va]%3)
        row["within_true_coarse_mean_accuracy"] = float(np.mean([v["val_accuracy"] for v in row["within_true_coarse"].values()]))
        results["representations"][key]=row
        save_json(HERE/"results.json",results)
        print(json.dumps(dict(representation=key, coarse3=row["coarse3"]["val_accuracy"],fine9=row["fine9"]["val_accuracy"],within_true_coarse=row["within_true_coarse_mean_accuracy"],elapsed_s=time.time()-started)),flush=True)


if __name__ == "__main__":
    main()
