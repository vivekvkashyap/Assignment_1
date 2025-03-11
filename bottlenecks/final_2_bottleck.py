import torch
from parler_tts import ParlerTTSForConditionalGeneration 
from transformers import AutoTokenizer
import soundfile as sf
import time

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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler_tts_mini_v0.1").to(device)
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler_tts_mini_v0.1")


    input_ids = tokenizer(descriptions, return_tensors="pt", padding="max_length", max_length=70).to(device)
    prompt_input_ids = tokenizer(prompts, return_tensors="pt", padding="max_length", max_length=70).to(device)

    generation = model.generate(input_ids=input_ids.input_ids,
                    attention_mask=input_ids.attention_mask,
                    prompt_input_ids=prompt_input_ids.input_ids,
                    prompt_attention_mask=prompt_input_ids.attention_mask)

    audio_arr = generation.cpu().numpy().squeeze()
    print('Done')
    # sf.write("parler_tts_out.wav", audio_arr, model.config.sampling_rate)

if __name__ == "__main__":
    main()


