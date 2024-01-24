import numpy as np
import torch
from pathlib import Path
import requests


from einops import rearrange

remote_model_path = (
    "https://example.com/valid_model_path.pt"
)


class ZoeDetector:
    def __init__(self):
        cwd = Path.cwd()
        ckpt_path = Path(cwd, "stencil_annotator")
        ckpt_path.mkdir(parents=True, exist_ok=True)
        modelpath = ckpt_path / "ZoeD_M12_N.pt"

        with requests.get(remote_model_path, stream=True) as r:
            r.raise_for_status()
            with open(modelpath, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        model = torch.hub.load(
            "monorimet/ZoeDepth",
            "ZoeD_N",
            pretrained=False,
            force_reload=False,
        )
        
        # Hack to fix the ZoeDepth import issue
        model_keys = model.state_dict().keys()
        loaded_dict = torch.load(modelpath, map_location=model.device)["model"]
        loaded_keys = loaded_dict.keys()
        for key in loaded_keys - model_keys:
            loaded_dict.pop(key)

        model.load_state_dict(loaded_dict)
        model.eval()
        self.model = model

    def __call__(self, input_image):
        assert input_image.ndim == 3
        image_depth = input_image
        with torch.no_grad():
            image_depth = torch.from_numpy(image_depth).float()
            image_depth = image_depth / 255.0
            image_depth = rearrange(image_depth, "h w c -> 1 c h w")
            depth = self.model.infer(image_depth)

            depth = depth[0, 0].cpu().numpy()

            vmin = np.percentile(depth, 2)
            vmax = np.percentile(depth, 85)

            depth -= vmin
            depth /= vmax - vmin
            depth = 1.0 - depth
            depth_image = (depth * 255.0).clip(0, 255).astype(np.uint8)

            return depth_image
