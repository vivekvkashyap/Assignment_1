import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import time
from accelerate import Accelerator

def main():

    prompts = [
        "Hello world.",
        "This is a test of the Parler TTS model.", 
        "The quick brown fox jumps over the lazy dog near the riverbank.",
        "Today we're going to look at how well this text-to-speech model performs with longer input text that requires more processing time.",
        "Natural language processing has advanced significantly in recent years. Text-to-speech systems like this one can now generate remarkably human-like voices with appropriate prosody and emotional expression. "
        "Today we're going to look at how well this text-to-speech model performs with longer input text that requires more processing time.",
        "Today we're going to look at how well this text-to-speech model performs with longer input text that requires more processing time.",
        "Today we're going to look at how well this text-to-speech model performs with longer input text that requires more processing time.",
        "Today we're going to look at how well this text-to-speech model performs with longer input text that requires more processing time.",
        "Today we're going to look at how well this text-to-speech model performs with longer input text that requires more processing time.",]
    
    description = ["A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."]

    accelerator = Accelerator()
    device = accelerator.device

    model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler_tts_mini_v0.1")
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler_tts_mini_v0.1")

    sampling_rate = model.config.sampling_rate
    model = accelerator.prepare(model)

    input_ids = tokenizer(description, return_tensors="pt", padding="max_length",max_length=80).to(device)
    prompt_input_ids = tokenizer(prompts, return_tensors="pt", padding="max_length",max_length=80).to(device)

    unwrapped_model = accelerator.unwrap_model(model)
    start_time = time.time()
    generation = unwrapped_model.generate(input_ids=input_ids.input_ids,
                attention_mask=input_ids.attention_mask,
                prompt_input_ids=prompt_input_ids.input_ids,
                prompt_attention_mask=prompt_input_ids.attention_mask,
                )
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

    # audio = generation.audio
    # sf.write("output.wav", audio, sampling_rate)

if __name__ == "__main__":
    main()

