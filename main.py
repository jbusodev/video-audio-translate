import os
import shutil
import sys
import subprocess  # cli commands for python
import argparse  # command line argument parser
import torch  # machine learning
import tempfile
from utils import env_vars

# Text
import whisper # speech recognition for transcription
from deepl import Translator  # translation

# Speech & Audio processing
from TTS.api import TTS  # text to speech
from pydub import (AudioSegment)  # audio manipulation
import pyrubberband as pyrb   # audio manipulation for pitch correction
import numpy as np   # mathematical operations on arrays of numbers

DEEPL_API_KEY = env_vars.DEEPL_API_KEY
LANG_DICT = {
    "en": "english",
    "es": "spanish",
    "fr": "french",
    "de": "german",
    "it": "italian",
    "ja": "japanese",
    "ko": "korean",
}


def extract_audio(video_path: str, output_audio_path: str, overwrite=False):
    """Extract audio from video file using ffmpeg."""
    command = ["ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", output_audio_path]
    if not os.path.exists(output_audio_path) or overwrite:
        subprocess.run(command, check=True)
        print(f"Audio successfully extracted.")
    else:
        print(f"File '{output_audio_path}' already exists. Skipping extraction.")


def transcribe_audio(input_audio_path: str):
    """Transcribe audio using whisper."""
    try:
        model = whisper.load_model("medium")
        result = model.transcribe(input_audio_path)
        print(f"File successfully transcribed.")
        return result["text"]
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return


def get_contents(input_file: str):
    """Returns the input file contents as a string."""
    with open(input_file, "r") as f:
        return f.read()


def translate_transcript(input_text: str, to_lang: str) -> str | list[str]:
    """Translate transcript to specified language using Deepl. Returns a string or list of strings."""
    translator = Translator(DEEPL_API_KEY)
    translation = translator.translate_text(input_text, target_lang=to_lang)
    return translation.text


def translate_audio(transcript: str, original_audio_path: str, translated_audio_path: str, language: str):
    """Translates audio using coquiTTS.
        Args:
            transcript (str) : Transcript to be generated.
            original_audio_path (str) : Path to input audio file.
            translated_audio_path (str) : Path to output audio file.
            language (str) : Language code of the target language.
    """
    # Checks for GPU availability for generation of audio
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{device} is available for TTS.")

    if not os.path.exists(translated_audio_path):
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        try:
            # Generate audio in chosen language from transcript and original audio.
            tts.tts_to_file(
                text=transcript,
                speaker_wav=original_audio_path,
                file_path=translated_audio_path,
                language=language,
            )
        except Exception as e:
            print(f"Error occurred during translation: {e}")
        else:
            print("Translation successful.")
    else:
        print(f"File '{translated_audio_path}' already exists. skipping translation.")


def match_audio_duration(original_audio_path: str, translated_audio_path: str):
    """Matches duration of translated with original audio. Overrides translated audio.
        Args:
            original_audio_path (str) : Path to original audio file.
            translated_audio_path (str) : Path to translated audio file.
    """
    # Load original and translated audio files
    original_audio = AudioSegment.from_file(original_audio_path)
    translated_audio = AudioSegment.from_file(translated_audio_path)
    
    # Get duration of audio files
    original_duration = len(original_audio)  # Duration in milliseconds
    translated_duration = len(translated_audio)  # Duration in milliseconds
    
    # Calculate the time stretch factor
    time_stretch_factor = original_duration / translated_duration
    
    # Convert translated audio to raw data
    translated_audio_array = np.array(translated_audio.get_array_of_samples())

    try:
        # Apply time-stretching
        stretched_audio_array = pyrb.time_stretch(translated_audio_array, translated_audio.frame_rate, time_stretch_factor)

        # Create a new AudioSegment from the time-stretched data
        stretched_audio = AudioSegment(
            stretched_audio_array.tobytes(),
            frame_rate=translated_audio.frame_rate,
            sample_width=translated_audio.sample_width,
            channels=translated_audio.channels
        )
        
        stretched_audio.export(translated_audio_path, format="wav")
        print(f"Time-stretching complete. Saved to '{translated_audio_path}'.")
    except Exception as e:
        print(f"Error while applying time-stretching. Error message: {e}")
        


def merge_audio_video(video_path: str, translated_audio_path: str, output_video_path: str):
    """Merge translated audio with original file using ffmpeg."""
    command = [
        "ffmpeg",
        "-i",
        video_path,
        "-i",
        translated_audio_path,
        "-c:v",
        "copy",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        output_video_path,
    ]
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Translate file audio to another language using coquiTTS."
    )
    parser.add_argument("file", type=str, help="Path to the input file")
    parser.add_argument(
        "--lang",
        "-l",
        type=str,
        required=True,
        help="Language code for translation (e.g., 'en', 'fr')",
    )
    parser.add_argument(
        "--merge",
        "-m",
        action="store_true",
        help="Merge translated audio with original video.",
    )

    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdirname:
        try:
            INPUT_FILE_PATH = os.path.abspath(args.file)

            # Exists script if source file is not found
            if not os.path.exists(INPUT_FILE_PATH):
                print(f"File does not exist. Please try with a different file.")
                sys.exit(0)
            
            LANG = args.lang
            BASE_NAME, EXTENSION = os.path.splitext(os.path.basename(INPUT_FILE_PATH))
            VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"]
            
            input_directory = os.path.dirname(INPUT_FILE_PATH)
            target_dir = tmpdirname if args.merge else input_directory
            
            # Output file generated by coquiTTS. - either in the same directory as.
            translated_audio_path = os.path.join(target_dir, f"{BASE_NAME}_{LANG}.wav")
            
            # Output translated video file
            output_video_path = os.path.splitext(args.file)[0] + f"_{LANG}.mp4"

            # Exists script if file already translated.
            if os.path.exists(translated_audio_path) or os.path.exists(output_video_path):
                print(
                    f"File has already been translated. Please try with a different file."
                )
                sys.exit(0)

            
            # if file is a video, extract audio
            if EXTENSION in VIDEO_EXTENSIONS:
                print(f"File is a video. Proceeding to extract audio...")
                original_audio_path = os.path.join(tmpdirname, "original_audio.wav")
                extract_audio(INPUT_FILE_PATH, original_audio_path)
            else:
                print(f"File is audio. Proceeding to transcription...")
                original_audio_path = INPUT_FILE_PATH  # audio file same as input file if input file is audio
                
            # Transcribes Audio then translates it.
            transcript = transcribe_audio(original_audio_path)
            translated_text = translate_transcript(transcript, to_lang=LANG)
            
            # Generate audio from text using extracted audio as speaker
            translate_audio(translated_text, original_audio_path, translated_audio_path, LANG)
            
            if args.merge:
                # Matches translated with original audio then merges with original video
                match_audio_duration(original_audio_path, translated_audio_path)
                merge_audio_video(args.file, translated_audio_path, output_video_path)
                print(f"Translated video saved to '{output_video_path}'")
            else:
                print(f"Translated audio saved to '{translated_audio_path}'")
        except Exception as e:
            print("Error:", str(e))
        finally:
            shutil.rmtree(tmpdirname)


if __name__ == "__main__":
    main()
