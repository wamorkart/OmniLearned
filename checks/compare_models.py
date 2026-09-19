import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

# Baseline small (single file, all events)
d_s = [np.load(f"/pscratch/sd/t/twamorka/omnilearned/eval/jetclass_distill/outputs_distill_jetclass_s_pretrain_l_a05_T4_100epochs_jetclass_test_rank{r}.npz") for r in range(16)]
probs_s = np.concatenate([x["prediction"][:, 1] for x in d_s])
labels_s = np.concatenate([x["pid"] for x in d_s])

# Distilled small (merge all rank files)
# d_l = [np.load(f"../outputs_fine_tune_top_l_top_{r}.npz") for r in range(16)]
# probs_l = np.concatenate([x["prediction"][:, 1] for x in d_l])
# labels_l = np.concatenate([x["pid"] for x in d_l])

for name, probs, labels in [
    ("Large finetuned", probs_s, labels_s),
    # ("Large finetuned", probs_l, labels_l),
]:
    auc = roc_auc_score(labels, probs)
    fpr, tpr, _ = roc_curve(labels, probs)
    for eff in [0.3, 0.5, 0.7]:
        idx = np.argmax(tpr >= eff)
        print(f"{name} | AUC={auc:.4f} | TPR={eff}: 1/FPR={1/fpr[idx]:.1f}")
        print(f"| AUC={auc:.4f} | TPR={eff}: 1/FPR={1/fpr[idx]:.1f}")        
