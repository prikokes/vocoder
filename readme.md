# Neural Audio Vocoders: Quality and Performance Evaluation

This repository contains the codebase for training, evaluating, and comparing various neural vocoder architectures. The main goal of this research is to analyze the trade-offs between audio synthesis quality, model size, and inference speed (Real-Time Factor — RTF).

Supported models include standard baselines and modern architectures such as **HiFi-GAN (V1 & V2)**, **FreeV**, and custom **iSTFT-based models** (Model 2: ISTFTWav, Model 3: ISTFTWavSnake).

## Project Architecture & Configuration

This project relies heavily on **[Hydra](https://hydra.cc/)** for flexible configuration management. Instead of hardcoding parameters, all settings for models, datasets, dataloaders, and training loops are modularized in the `src/configs/` directory.

The entry point for the configuration is the **`baseline.yaml`** file. You can easily switch between models or hyperparameters by overriding values in the command line or creating new composition configs (e.g., `istftwav.yaml`).

In our study, we compared heavy baseline models against lightweight and fast architectures. The evaluation was conducted across two main dimensions: synthesis quality (PESQ, STOI, LSD, Mel Distance) and computational complexity (parameters, MACs, RTF on CPU/GPU). All compared models were retrained from scratch under an identical 48-hour training budget to ensure a fair comparison.

## Research Results

### 1. Audio Synthesis Quality

Mean values are reported with 95% confidence intervals computed via bootstrap over the LJSpeech test split.

| Model | PESQ ↑ | STOI ↑ | LSD ↓ | Mel Dist. ↓ |
| :--- | :---: | :---: | :---: | :---: |
| HiFi-GAN V1 | 3.244 ± 0.026 | 0.960 ± 0.003 | 1.245 ± 0.005 | 0.171 ± 0.001 |
| HiFi-GAN V2 | 2.449 ± 0.023 | 0.928 ± 0.003 | 1.662 ± 0.007 | 0.226 ± 0.001 |
| FreeV | **3.690 ± 0.027** | **0.975 ± 0.003** | 0.718 ± 0.003 | **0.168 ± 0.001** |
| Model 2 | 3.055 ± 0.027 | 0.950 ± 0.003 | 0.734 ± 0.004 | 0.209 ± 0.001 |
| Model 3 | 3.317 ± 0.026 | 0.959 ± 0.003 | **0.733 ± 0.004** | 0.202 ± 0.001 |

### 2. Performance & Computational Complexity

MACs are reported per one second of generated audio at 22050 Hz. RTF is measured in offline mode on full sequences; values below 1.0 indicate generation faster than real time.

| Model | Params (M) ↓ | MACs (G) ↓ | RTF CPU ↓ | RTF GPU ↓ |
| :--- | :---: | :---: | :---: | :---: |
| HiFi-GAN V1 | 13.90 | 30.82 | 0.0762 | 0.1052 |
| HiFi-GAN V2 | **0.93** | 1.91 | 0.0196 | 0.0431 |
| FreeV | 18.20 | 1.56 | **0.0040** | 0.0129 |
| Vocos | 13.50 | 1.17 | — | — |
| Model 2 | 2.41 | 0.977 | 0.0070 | **0.0119** |
| Model 3 | 2.39 | **0.860** | 0.0104 | 0.0175 |

### Summary

- **Model 3** matches FreeV on LSD while being **~7.6× smaller** in parameter count and **~1.8× cheaper** in MACs.
- **Model 2** delivers the lowest GPU RTF among all evaluated models and the lowest first-chunk latency in streaming mode.
- Both proposed models stay within a single confidence interval of HiFi-GAN V1 on STOI and outperform HiFi-GAN V2 across every quality metric while being ~3× smaller than V1.

## Training

To train the models from scratch, run the `train.py` script and specify the model architecture. You can also load your environment variables via a `.env` file (e.g., `COMET_API_KEY`).

```bash
python train.py model=hifigan
python train.py model=freev
python train.py model=istftwav
```

## Running inference

```
python inference.py model=freev
```

There also provided scripts for model evaluations in eval directory, you can run (configs for this scripts are hardcoded in the scripts).

```
eval_open_models.py - is used to eval quality of models that alreeady exist.
eval_repro.py - was used to train models that were retrained in this pipeline.
performance_metrics.py - was used for evaluating performance metrics of models that alreeady exist.
quality_ci.py - was used to built confidence intervals for quality metrics.
stat_tests.py - was used to run stat tests on RTF streming metrics.
streaming_eval.py - was used to evaluate streaming RTF on models that alreeady exist.
streaming_metrics.py - was used to evaluate streaming metrics on trained models.
```

## License
This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.