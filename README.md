# Sarvam Assignment Repository
Below is the repository for submitting the assignment on optimizing autoregressive models, with a focus on Parler TTS models.

## 📖 Quick Index
* Identifying bottlenecks in Parler TTS models
* Latency optimization for text-to-speech models
* Throughput optimization
* Real-time streaming inference

### Repository Structure
    .
    ├── additional_results   # Additional results for the assignment
    │   └── miscellaneous.ipynb
    ├── bottlenecks          # Bottlenecks for the assignment
    │   ├── final_1_bottleck.ipynb
    │   ├── final_2_bottleck.py
    │   ├── tts_profile.qdrep
    │   ├── tts_profile.qdstrm
    │   └── tts_profile.sqlite
    ├── latency              # Latency for the assignment
    │   ├── final_latency_1.ipynb
    │   └── final_latency_3.ipynb
    ├── streaming            # Streaming for the assignment
    │   └── Final_Streaming.ipynb
    ├── throughput           # Throughput for the assignment
    │   └── acc_parler.py
    ├── README.md
    └── requirements.txt

## 📦 GPU Used

I have used the following GPUs for this assignment:

### GPU Specifications
* 2x NVIDIA RTX A6000
  * Driver Version: 560.35.03
  * CUDA Version: 12.6
  * Memory: 48GB GDDR6 (49140MiB)
  * Maximum Power Consumption: 300W


## 📦 Installation

I have used the following dependencies for this assignment:

Note: Additional dependencies were already installed in the environment, as I used a previous Conda environment and had previously downloaded the necessary dependencies for the assignment.

```bash
pip install -r requirements.txt
```

## 📝 Usage

* Jupyter notebooks can be run and do not require any additional files.
* .py files require specific commands for execution, which are mentioned within each respective file.

## 📝 Additional Results

* Additional results for the assignment are present in the `additional_results` folder.
* The `miscellaneous.ipynb` file contains additional results for the assignment.