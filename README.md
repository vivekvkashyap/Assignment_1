# Sarvam Assignment Repository
Below is the repository for submitting the assignment on optimizing autoregressive models, with a focus on Parler TTS models.

## 📖 Quick Index
* Identifying bottlenecks in Parler TTS models
* Latency optimization for text-to-speech models
* Throughput optimization
* Real-time streaming inference


## Repository Structure

📦 Assignment_1
├── 📂 bottlenecks               # Contains datasets and preprocessed data
│   ├── final_1_bottleneck.ipynb # Raw datasets
│   ├── final_2_bottleneck.ipynb # Preprocessed data
│   ├── tts_profile.qdrep       # Instructions for dataset usage
│   ├── tts_profile.qdstrm      # Preprocessed data
│   └── tts_profile.sqlite      # Instructions for dataset usage
├── 📂 models                    # Pretrained and optimized model checkpoints
│   ├── baseline/               # Unoptimized models
│   ├── optimized/              # Optimized models
│   └── README.md               # Model details
├── 📂 scripts                   # Python scripts for training and optimization
│   ├── train.py                # Training script for TTS models
│   ├── infer.py                # Inference and evaluation script
│   ├── optimize.py             # Optimization and quantization script
│   └── utils.py                # Helper functions
├── 📂 notebooks                 # Jupyter notebooks for experiments
│   ├── exploratory.ipynb       # Data exploration and bottleneck analysis
│   ├── latency_tests.ipynb     # Latency optimization experiments
│   └── throughput_tests.ipynb  # Throughput optimization experiments
├── 📂 configs                   # Configuration files
│   ├── model_config.yaml       # Model hyperparameters
│   ├── triton_config.pbtxt     # Triton server configurations
│   └── README.md               # Configuration details
├── 📂 results                   # Benchmarking results and logs
│   ├── latency/                # Latency test results
│   ├── throughput/             # Throughput analysis
│   ├── streaming/              # Streaming optimization results
│   └── README.md               # Summary of results
├── requirements.txt            # Dependencies and package requirements
├── setup.py                    # Installation script (if packaging required)
├── README.md                   # Project documentation
└── LICENSE                     # License information
