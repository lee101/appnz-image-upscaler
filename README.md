# appnz-image-upscaler

[![Deploy to app.nz](https://app.nz/deploy-button.svg)](https://app.nz/deploy?image=ghcr.io/lee101/appnz-image-upscaler:latest&name=image-upscaler&vram=8)

[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) 4x image upscaling with
optional [GFPGAN](https://github.com/TencentARC/GFPGAN) face restoration,
packaged as an [app.nz cog](https://app.nz): a tiny HTTP contract on port 5000
with `POST /predictions` in and a PNG data URI out. Runs on CPU; uses CUDA
(fp16 + tiling) when a GPU is present. RealESRGAN_x4plus, GFPGANv1.4 and
facexlib weights are baked into the image for fast cold starts.

## Inputs

| name | type | notes |
|---|---|---|
| `image` | image | https URL or `data:` URI |
| `scale` | enum | `2` or `4` (default `4`) |
| `face_enhance` | boolean | restore faces with GFPGAN (default `false`) |

Output: `data:image/png;base64,...`.

## Run locally

```bash
docker run -p 5000:5000 ghcr.io/lee101/appnz-image-upscaler:latest

curl -s http://localhost:5000/health-check

curl -s http://localhost:5000/predictions -X POST \
  -H 'Content-Type: application/json' \
  -d '{"input": {"image": "https://example.com/photo.jpg", "scale": 4, "face_enhance": true}}' \
  | python3 -c 'import sys,json,base64; open("out.png","wb").write(base64.b64decode(json.load(sys.stdin)["output"].split(",",1)[1]))'
```

## One-click deploy on app.nz

Click the badge above, or open
`https://app.nz/deploy?image=ghcr.io/lee101/appnz-image-upscaler:latest&name=image-upscaler&vram=8`.

## Version pins

basicsr 1.4.2 + realesrgan 0.3.0 break on newer torchvision (the removed
`torchvision.transforms.functional_tensor` module) and numpy 2, so the image
pins torch 2.1.2 / torchvision 0.16.2 (CPU wheels), numpy<2, and patches the
import in the Dockerfile.

## Build

```bash
docker build -t ghcr.io/lee101/appnz-image-upscaler:latest .
```

GitHub Actions builds and pushes `ghcr.io/lee101/appnz-image-upscaler:latest`
on every push to `main`.

## License

MIT
