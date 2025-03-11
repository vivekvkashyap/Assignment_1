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

## Models Used

I have used only one model for this assignment for the sake of simplicity.

* [Parler TTS_mini_v0.1](https://huggingface.co/parler-tts/parler_tts_mini_v0.1) : an 880M parameter model.

## Data Used

I have used the following data for this assignments. Consists of short sentences and long sentences.

```bash
prompts = [
    "Hello world.",
    "A futuristic AI assistant responding in a clear, robotic yet friendly manner",
    "A dramatic voice-over for an action-packed movie trailer that builds suspense",
    "A casual and friendly conversation starter that feels natural and engaging",
    "An enchanting introduction to a bedtime story that sparks imagination",
    "A detailed weather update that provides temperature, wind speed, and overall forecast",
    "A robotic voice delivering a monotone yet precise system status update",
    "A powerful and energetic motivational speech that inspires action and confidence"
]

descriptions = [
    "A female speaker delivers a warm and welcoming message with a slightly expressive and friendly tone. The speech has a moderate pace and a natural intonation, making it feel inviting. The recording is clear, with minimal background noise.",
    "A robotic AI voice speaks in a neutral yet slightly friendly manner. The speech is steady, with a consistent pitch and minimal variation in tone. The audio is crisp, resembling a synthesized assistant's response.",
    "A deep male voice narrates with a dramatic and intense tone, building suspense with pauses and rising intensity. The speech is slow-paced, emphasizing key moments. The recording is cinematic, with a slight reverberation for a grand effect.",
    "A young adult male speaker talks in a casual and relaxed tone. The speech is natural, with slight variations in pitch and pauses that mimic real-life conversations. The recording is high quality, making it feel like a personal chat.",
    "A soft-spoken female speaker introduces a bedtime story with a soothing and melodic tone. The pace is slow and gentle, making it easy to follow. The recording has a slight warmth, resembling a close-up microphone capture.",
    "A professional male voice delivers a clear and informative weather update. The tone is neutral but engaging, with a moderate pace. The articulation is precise, and the recording is high quality, with no distortions.",
    "A synthetic robotic voice speaks in a completely monotone and even-paced manner. The pitch remains constant, with no emotional inflection. The recording is crisp and clean, resembling an automated system response.",
    "A dynamic and energetic male speaker delivers an inspiring speech with a strong, enthusiastic tone. The pace is varied, with emphasis on key words to motivate the listener. The recording is sharp and immersive, with no background noise."
]

```



## 📝 Usage

* Jupyter notebooks can be run and do not require any additional files.

For .py files : 
File Name : `acc_parler.py`

```bash
accelerate launch --num_processes 2 acc_parler.py
```

File Name : `final_2_bottleck.py`

```bash
python final_2_bottleck.py
```



## 📝 Additional Results

* Additional results for the assignment are present in the `additional_results` folder.
* The `miscellaneous.ipynb` file contains additional results for the assignment.