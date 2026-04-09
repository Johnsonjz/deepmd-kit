import torch
import deepmd.pt

try:
    print(torch.ops.deepmd.nufft_sog_frame_correction_bundle)
    print("Success: Operator loaded!")
except Exception as e:
    print("Error:", e)
