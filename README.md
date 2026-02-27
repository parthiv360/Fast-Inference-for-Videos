# Efficient Video Summarization Pipeline using SmolVLM2

This repository contains an end-to-end, highly optimized machine learning pipeline for processing and summarizing short-form video content. Utilizing the **SmolVLM2-2.2B-Instruct** Vision Language Model (VLM), the system ingests video files and audio transcripts to generate grounded, hallucination-free summaries and accurately categorize the content. This project was part of the seminar **Efficient Training of Large Language Models**.

A major focus of this project is **inference optimization**, benchmarking a baseline model against various acceleration techniques to reduce execution time while preserving summary quality (measured via BERTScore) and categorization accuracy.

## 🚀 Key Features

* **End-to-End Processing:** Seamlessly handles video loading, audio transcription extraction, prompt formatting, VLM text generation, and post-processing.
* **Automated Evaluation:** Built-in benchmarking against ground-truth data using **BERTScore** for summary quality and **Accuracy** for category prediction.
* **Inference Profiling:** Granular time-tracking for each pipeline component (loading, transcription, generation) to identify bottlenecks.
* **Optimization Strategies:** Implements advanced ML optimization techniques to reduce total inference time, including:
    * Model quantization (bitsandbytes)
    * Data Parallelism
    * Batch processing and parallel data loading
    * Mixed Precision Training



## 🧠 Model & Dataset

* **Model:** [SmolVLM2-2.2B-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct) – an open-source, efficient Vision Language Model.
* **Dataset:** 270 short-form videos (ranging from 5 seconds to 5 minutes) in `.mp4` format, accompanied by audio transcripts in `.txt` format.

## 📂 Project Structure

```text
├── data/
│   ├── inputs/
│   │   ├── videos/              # Place .mp4 files here
│   │   └── audio_transcripts/   # Place .txt transcripts here
│   └── outputs/                 # CSV reports and generated summaries saved here
├── src/
│   ├── config.py                # Pipeline configuration and path management
│   ├── main.py                  # Main execution script
│   └── evaluate.py              # BERTScore and Accuracy calculation logic
├── docs/
│   └── project_report.pdf       # Detailed analysis of baseline vs. optimized performance
├── requirements.txt             # Project dependencies
└── README.md
```

## ⚙️ Installation & Setup

**Prerequisites:** Python 3.11 is strictly required for dependency compatibility. 

**1. Clone the repository:**
```bash
git clone
cd video-summarization-pipeline
```

**2. Create and activate a virtual environment:**
```bash
python3.11 -m venv venv
# On macOS/Linux
source venv/bin/activate  
# On Windows
venv\Scripts\activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```
📢 **Note:** The complete output files containing baseline and optimized performance logs, as well as the detailed 2-3 page optimization report, can be found in [this Google Drive link](https://drive.google.com/drive/folders/1trqemP04iJrsRHYE1UyaFizcaWOVQzWY?usp=drive_link).
