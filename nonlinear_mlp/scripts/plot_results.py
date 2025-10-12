import json
import glob
import os
import matplotlib.pyplot as plt

def load_histories(path="runs"):
    data = []
    for run_dir in glob.glob(f"{path}/*"):
        hist_path = os.path.join(run_dir, "history.json")
        meta_path = os.path.join(run_dir, "meta.json")
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                hist = json.load(f)
            with open(meta_path) as f:
                meta = json.load(f)
            final = hist[-1]
            run_name = run_dir.split("/")[-1]
            # attempt parse ratio from name
            ratio = None
            for token in run_name.split("_"):
                if token.isdigit():
                    val = int(token)
                    if 0 <= val <= 100:
                        ratio = val / 100.0
            data.append({"run": run_name, "ratio": ratio, "final_acc": final["val_acc"], "meta": meta})
    return data

def main():
    data = load_histories()
    data = [d for d in data if d["ratio"] is not None]
    data.sort(key=lambda x: x["ratio"])
    xs = [d["ratio"] * 100 for d in data]
    ys = [d["final_acc"] for d in data]
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel("% Linear Neurons")
    plt.ylabel("Validation Accuracy")
    plt.title("Accuracy vs Linear Neuron Percentage")
    plt.grid(True)
    plt.savefig("accuracy_vs_linear.png")
    print("Saved accuracy_vs_linear.png")

if __name__ == "__main__":
    main()