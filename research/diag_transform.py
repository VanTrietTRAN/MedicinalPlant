"""Kiểm giả thuyết: số 85.27/82.79 trong bài sinh ra từ transform sai?"""
import sys, numpy as np, torch, timm
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import evaluate_models as EM
from torchvision import transforms
from torch.utils.data import DataLoader


def main():
    samples, classes = EM.make_index(r"D:/datasets/medical-split-fn/test")
    labels = np.array([s[1] for s in samples])
    torch.set_num_threads(12)
    print(f"test: {len(samples)} ảnh, {len(classes)} lớp\n")
    for name in ["ConvNeXt-Base", "Swin-Base"]:
        spec = EM.MODEL_SPECS[name]
        m = timm.create_model(spec["timm_name"], pretrained=False, num_classes=206)
        m.load_state_dict(torch.load(spec["ckpt"], map_location="cpu", weights_only=True))
        m.eval()
        for tfname in ["resize224", "resize256_crop224"]:
            dl = DataLoader(EM.EvalDS(samples, EM.build_transform(tfname, transforms)),
                            batch_size=32, num_workers=4)
            preds = []
            with torch.no_grad():
                for x, _ in dl:
                    preds.append(m(x).argmax(1).numpy())
            acc = (np.concatenate(preds) == labels).mean() * 100
            tag = "  <-- transform lúc train" if tfname == spec["eval_tf"] else ""
            print(f"{name:16s} {tfname:20s} Top-1 = {acc:6.2f}%{tag}", flush=True)
        print()


if __name__ == "__main__":
    main()
