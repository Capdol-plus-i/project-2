# one-off_convert_ckpt.py
import torch, sys
path = sys.argv[1] if len(sys.argv) > 1 else "residual_nan_signal_mlp.pth"
ckpt = torch.load(path, map_location="cpu")
if "model_state_dict" not in ckpt and "state_dict" in ckpt:
    ckpt["model_state_dict"] = ckpt["state_dict"]
    torch.save(ckpt, path)
    print("✅ converted:", path)
else:
    print("ℹ️ no change needed")
