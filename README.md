<div align="center">

# FramePack-ZLUDA

FramePack ZLUDA

</div>

## Install

- install python from [www.python.org](https://www.python.org/) .
   Tested Python3.10.17 and Python3.11.12 .
- install [git](https://git-scm.com/).
- install [AMD HIP SDK for Windows 6.2.4](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html) .
   If  5.7.1 already installed, uninstall 5.7.1 before installing 6.2.4 .
- Open Command Prompt (not PowerShell), then run the following:
```
   git clone https://github.com/githubcto/FramePack-ZLUDA.git
   cd FramePack-ZLUDA
   python.exe -m venv venv
   venv\Scripts\activate.bat
   python.exe -m pip install --upgrade pip
   pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install -r requirements.txt
   curl -s -L https://github.com/lshqqytiger/ZLUDA/releases/download/rel.dba64c0966df2c71e82255e942c96e2e1cea3a2d/ZLUDA-windows-rocm6-amd64.zip > zluda.zip
   mkdir .zluda && tar -xf zluda.zip -C .zluda  --strip-components=1
```
   Next, replace the following DLL files in venv/Lib/site-packages/torch/lib with the ones provided in the .zluda folder:
   
   cublas.dll
   cusparse.dll
   cufft.dll
   cufftw.dll
   nvrtc.dll
   
   You can find them at venv/Lib/site-packages/torch/lib.
   
   or,
   
   [download zip](https://github.com/githubcto/FramePack-ZLUDA/archive/refs/heads/main.zip), extract, and refer to install-win-zluda.bat .



## Run
FramePack-user.bat

1st run,
- will download 40GB.
- ZLUDA compile takes 30 minutes or more.

   Duriing this 30 minutes, you'll see the message 

   "Compilation is in progress. Please wait..." every minute.

If FramePack-user.bat does not work, try FramePack-user-DEVICE0.bat or FramePack-user-DEVICE1.bat .

## Tips
1st time generate,
 try
- Use square image. FramePack read it and resize automatically.
- 1 sec
- 10 steps
- other values: use preset
- Start Generation, see VRAM and DRAM usage.

DRAM 64GB minimum. 64GB enough for linux. 96GB enough for windows. 128GB recommend.

Set windowOS page file "auto", "64GB" or more.

TeaCache is fast, but output quality is not so good. Try TeaCache and you feel good movie, then disable TeaCache and try same seed again.

Saved png files can be converted mp4 movie [using ffmpeg](https://ffmpeg.org/) like this.
```
ffmpeg.exe -framerate 30 -i %4d.png -c:v libx264 -crf 23 -pix_fmt yuv420p -an out.mp4
```

## Issue
Since my VGA is RX 6000, I can not verify some Attentions which RX 7000 support, for example,

[Repeerc/flash-attention-v2-RDNA3-minimal](https://github.com/Repeerc/flash-attention-v2-RDNA3-minimal)

( You need modify [demo_gradio.py (around line 10th.)](https://github.com/githubcto/FramePack-ZLUDA/blob/main/demo_gradio.py#L10) from "False" to "True", maybe.)

torch.backends.cuda.enable_flash_sdp(False)

.

So, I shall close FramePack-ZLUDA repo without any notice.

## for Linux ROCm

This code may help.

FramePack-ZLUDA [demo_gradio.py](https://github.com/githubcto/FramePack-ZLUDA/blob/main/demo_gradio.py#L74-L87) 
```
# VAE Tiling size
vae.enable_tiling(
    tile_sample_min_height=128,  #256
    tile_sample_min_width=128,   #256
    tile_sample_min_num_frames=12,  #16
    tile_sample_stride_height=96,  #292
    tile_sample_stride_width=96,   #192
    tile_sample_stride_num_frames=10   #12
)
```

VAE tile size is adjustable. not only resolution but also frames.

source code

venv/Lib/site-packages/diffusers/models/autoencoders/autoencoder_kl_hunyuan_video.py

HuggingFace diffusers [autoencoder_kl_hunyuan_video.py](https://github.com/huggingface/diffusers/blob/f00a995753732210a696de447cd0db80e181c30a/src/diffusers/models/autoencoders/autoencoder_kl_hunyuan_video.py#L717-L766) 

## ChangeLog

2025 Apr. 26th : add FPS switch. default=24fps. QuickList2nd changed. (torch2.7.0 and ZLUDA3.9.3 works, but keep torch2.6.0 for a while).

2025 Apr. 25th : Init. ZLUDA, RESOLUTION, SAVE PNG, README.

---
---

<p align="center">

</p>

# FramePack

Official implementation and desktop software for ["Packing Input Frame Context in Next-Frame Prediction Models for Video Generation"](https://lllyasviel.github.io/frame_pack_gitpage/).

Links: [**Paper**](https://arxiv.org/abs/2504.12626), [**Project Page**](https://lllyasviel.github.io/frame_pack_gitpage/)

FramePack is a next-frame (next-frame-section) prediction neural network structure that generates videos progressively. 

FramePack compresses input contexts to a constant length so that the generation workload is invariant to video length.

FramePack can process a very large number of frames with 13B models even on laptop GPUs.

FramePack can be trained with a much larger batch size, similar to the batch size for image diffusion training.

**Video diffusion, but feels like image diffusion.**

# News

**2025 May 03:** The FramePack-F1 is released. [Try it here.](https://github.com/lllyasviel/FramePack/discussions/459)

Note that this GitHub repository is the only official FramePack website. We do not have any web services. All other websites are spam and fake, including but not limited to `framepack.co`, `frame_pack.co`, `framepack.net`, `frame_pack.net`, `framepack.ai`, `frame_pack.ai`, `framepack.pro`, `frame_pack.pro`, `framepack.cc`, `frame_pack.cc`,`framepackai.co`, `frame_pack_ai.co`, `framepackai.net`, `frame_pack_ai.net`, `framepackai.pro`, `frame_pack_ai.pro`, `framepackai.cc`, `frame_pack_ai.cc`, and so on. Again, they are all spam and fake. **Do not pay money or download files from any of those websites.**

# Requirements

Note that this repo is a functional desktop software with minimal standalone high-quality sampling system and memory management.

**Start with this repo before you try anything else!**

Requirements:

* Nvidia GPU in RTX 30XX, 40XX, 50XX series that supports fp16 and bf16. The GTX 10XX/20XX are not tested.
* Linux or Windows operating system.
* At least 6GB GPU memory.

To generate 1-minute video (60 seconds) at 30fps (1800 frames) using 13B model, the minimal required GPU memory is 6GB. (Yes 6 GB, not a typo. Laptop GPUs are okay.)

About speed, on my RTX 4090 desktop it generates at a speed of 2.5 seconds/frame (unoptimized) or 1.5 seconds/frame (teacache). On my laptops like 3070ti laptop or 3060 laptop, it is about 4x to 8x slower. [Troubleshoot if your speed is much slower than this.](https://github.com/lllyasviel/FramePack/issues/151#issuecomment-2817054649)

In any case, you will directly see the generated frames since it is next-frame(-section) prediction. So you will get lots of visual feedback before the entire video is generated.

# Installation

**Windows**:

[>>> Click Here to Download One-Click Package (CUDA 12.6 + Pytorch 2.6) <<<](https://github.com/lllyasviel/FramePack/releases/download/windows/framepack_cu126_torch26.7z)

After you download, you uncompress, use `update.bat` to update, and use `run.bat` to run.

Note that running `update.bat` is important, otherwise you may be using a previous version with potential bugs unfixed.


Note that the models will be downloaded automatically. You will download more than 30GB from HuggingFace.

**Linux**:

We recommend having an independent Python 3.10.

    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    pip install -r requirements.txt

To start the GUI, run:

    python demo_gradio.py

Note that it supports `--share`, `--port`, `--server`, and so on.

The software supports PyTorch attention, xformers, flash-attn, sage-attention. By default, it will just use PyTorch attention. You can install those attention kernels if you know how. 

For example, to install sage-attention (linux):

    pip install sageattention==1.0.6

However, you are highly recommended to first try without sage-attention since it will influence results, though the influence is minimal.

# GUI

# Sanity Check

Before trying your own inputs, we highly recommend going through the sanity check to find out if any hardware or software went wrong. 

Next-frame-section prediction models are very sensitive to subtle differences in noise and hardware. Usually, people will get slightly different results on different devices, but the results should look overall similar. In some cases, if possible, you'll get exactly the same results.

---

# Prompting Guideline

Many people would ask how to write better prompts. 

Below is a ChatGPT template that I personally often use to get prompts:

    You are an assistant that writes short, motion-focused prompts for animating images.

    When the user sends an image, respond with a single, concise prompt describing visual motion (such as human activity, moving objects, or camera movements). Focus only on how the scene could come alive and become dynamic using brief phrases.

    Larger and more dynamic motions (like dancing, jumping, running, etc.) are preferred over smaller or more subtle ones (like standing still, sitting, etc.).

    Describe subject, then motion, then other things. For example: "The girl dances gracefully, with clear movements, full of charm."

    If there is something that can dance (like a man, girl, robot, etc.), then prefer to describe it as dancing.

    Stay in a loop: one image in, one motion prompt out. Do not explain, ask questions, or generate multiple options.



*The man dances powerfully, striking sharp poses and gliding smoothly across the reflective floor.*

Usually this will give you a prompt that works well. 

You can also write prompts yourself. Concise prompts are usually preferred, for example:

*The girl dances gracefully, with clear movements, full of charm.*

*The man dances powerfully, with clear movements, full of energy.*

and so on.

# Cite

    @article{zhang2025framepack,
        title={Packing Input Frame Contexts in Next-Frame Prediction Models for Video Generation},
        author={Lvmin Zhang and Maneesh Agrawala},
        journal={Arxiv},
        year={2025}
    }
