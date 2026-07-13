"""Blackwell-compatible Real-ESRGAN Cog adapter with optional GFPGAN."""

from pathlib import Path as LocalPath

from cog import BaseRunner, Input, Path


MODEL_SPECS = {
    "general": {"file": "RealESRGAN_x4plus.pth", "blocks": 23},
    "anime": {"file": "RealESRGAN_x4plus_anime_6B.pth", "blocks": 6},
}


def validated_scale(scale: int) -> int:
    if scale not in (2, 4):
        raise ValueError("scale must be 2 or 4")
    return scale


def destination_for(source: str, has_alpha: bool) -> LocalPath:
    suffix = ".png" if has_alpha else ".jpg"
    return LocalPath("/tmp") / f"{LocalPath(source).stem}-upscaled{suffix}"


class Runner(BaseRunner):
    def setup(self) -> None:
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        self._torch = torch
        self._upsamplers = {}
        for name, spec in MODEL_SPECS.items():
            network = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=spec["blocks"],
                num_grow_ch=32,
                scale=4,
            )
            self._upsamplers[name] = RealESRGANer(
                scale=4,
                model_path=f"/weights/{spec['file']}",
                model=network,
                tile=256,
                tile_pad=16,
                pre_pad=0,
                half=torch.cuda.is_available(),
            )
        self._faces = {}

    def _face_enhancer(self, model: str, scale: int):
        from gfpgan import GFPGANer

        key = (model, scale)
        if key not in self._faces:
            self._faces[key] = GFPGANer(
                model_path="/weights/GFPGANv1.3.pth",
                upscale=scale,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=self._upsamplers[model],
            )
        return self._faces[key]

    def run(
        self,
        image: Path = Input(description="PNG, JPEG, or WebP image"),
        scale: int = Input(description="Output scale", default=4, choices=[2, 4]),
        model: str = Input(description="Restoration model", default="general", choices=["general", "anime"]),
        face_enhance: bool = Input(description="Restore faces with GFPGAN", default=False),
    ) -> Path:
        import cv2

        scale = validated_scale(scale)
        if model not in MODEL_SPECS:
            raise ValueError(f"unsupported model: {model}")
        source = str(image)
        pixels = cv2.imread(source, cv2.IMREAD_UNCHANGED)
        if pixels is None:
            raise ValueError("input is not a readable image")
        has_alpha = len(pixels.shape) == 3 and pixels.shape[2] == 4
        if face_enhance:
            _, _, output = self._face_enhancer(model, scale).enhance(
                pixels, has_aligned=False, only_center_face=False, paste_back=True
            )
        else:
            output, _ = self._upsamplers[model].enhance(pixels, outscale=scale)
        destination = destination_for(source, has_alpha)
        params = [cv2.IMWRITE_JPEG_QUALITY, 95] if destination.suffix == ".jpg" else []
        if not cv2.imwrite(str(destination), output, params):
            raise RuntimeError("could not encode output image")
        return Path(destination)
