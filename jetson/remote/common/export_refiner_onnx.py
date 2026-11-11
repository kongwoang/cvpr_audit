import os, sys, torch
sys.path.insert(0, os.environ["B"])
import train_refiner_remote as TR
ck = torch.load(f"{os.environ['B']}/out/resnet18/image_refiner.pt", map_location="cpu", weights_only=False)
net = TR.Res18(len(ck["fmu"])); net.load_state_dict(ck["state"]); net.eval()
x, f = torch.zeros(1, 3, 64, 64), torch.zeros(1, len(ck["fmu"]))
torch.onnx.export(net, (x, f), f"{os.environ['B']}/out/resnet18/image_refiner.onnx", input_names=["crop", "feats"],
                  output_names=["log_ratio"], opset_version=13, dynamic_axes=None)
import numpy as np; np.savez(f"{os.environ['B']}/out/resnet18/norm.npz", fmu=ck["fmu"], fsd=ck["fsd"])
print("exported")
