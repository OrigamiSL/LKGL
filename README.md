# LKGL: Enhancing Cross-View Geo-localization via Global--Local Feature Matching and Keypoint Guidance

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/OrigamiSL/LKGL?style=social)](https://github.com/OrigamiSL/LKGL)

The official PyTorch implementation of **LKGL** (Local-Keypoint-Guided Localization). 

## 📖 Abstract
Cross-view geo-localization (CVGL), which aims to match ground- or drone-view imagery with satellite maps, has emerged as a fundamental task in remote sensing, with broad applications in urban monitoring and unmanned aerial vehicle (UAV) localization. Despite remarkable progress, existing approaches rely heavily on metric-based comparisons of global image features, which often leads to mismatches in regions with highly similar global appearances but distinct local structures. To address this limitation, this work introduces a novel CVGL framework, termed LKGL, which integrates two complementary methods. First, the global--local feature matching (GLFM) method computes similarity at both holistic and localized levels and integrates a reliability-based filtering mechanism to remove ambiguous or low-confidence local correspondences, thereby enhancing robustness against confusing global patterns. Second, the keypoint-guided feature pyramid network (KGFPN) integrates multiscale features with spatial priors derived from keypoint maps through attention-based interactions between shallow and deep layers. This design further enhances local feature expressiveness and structural sensitivity. Together, the GLFM and KGFPN enable LKGL to balance global contextual information with fine-grained local cues, achieving state-of-the-art performance on University-1652 and SUES-200. The source code is released on [https://github.com/OrigamiSL/LKGL](https://github.com/OrigamiSL/LKGL).

## 🖼️ Architecture
![Architecture Overview](img/architecture.png)

## ⚙️ Environment Setup

You can easily set up the environment using Conda and the provided `LKGL.yml` file.

```bash
conda env create -f LKGL.yml
conda activate pb_reid
