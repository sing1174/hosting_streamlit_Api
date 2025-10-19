import streamlit as st
import os
import json
import yt_dlp
import whisper_timestamped
import numpy as np
from pydub import AudioSegment
from moviepy import *
#from moviepy.editor import VideoFileClip
import shutil
import subprocess
import re
import time

# Helper functions
def download_video_from_youtube(url):
    try:
        ydl_opts = {
            'format': 'bestvideo+bestaudio/best',
            'outtmpl': 'downloaded_video.%(ext)s',
            'merge_output_format': 'mp4',
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            video_file = "downloaded_video.mp4"
            return video_file
    except Exception as e:
        return str(e)

def download_audio_from_youtube(url):
    try:
        # Options to download audio and video
        ydl_opts = {
            'format': 'bestaudio/best',  # Downloads the best quality audio
            'outtmpl': '%(id)s.%(ext)s',  # Saves the file with YouTube video ID as the name
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',  # Use FFmpeg to extract audio
                'preferredcodec': 'wav',  # Desired audio format
                'preferredquality': '192',  # Audio quality
            }],
            'noplaylist': True,  # Avoid downloading playlists
            'extractaudio': True,  # Ensures only audio is extracted if no video is needed
        }

        # Using yt-dlp to download and extract audio
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            audio_file = f"{info_dict['id']}.wav"  # Save as wav
            print(f"Audio file saved: {audio_file}")
            return audio_file  # Returns the path of the audio file
    except Exception as e:
        print(f"Error: {e}")
        return str(e)

def extract_audio_from_video(video_file):
    try:
        audio_file = "extracted_audio.wav"
        video_clip = VideoFileClip(video_file)
        video_clip.audio.write_audiofile(audio_file, codec='pcm_s16le', ffmpeg_params=["-ac", "1", "-ar", "16000"])
        return audio_file
    except Exception as e:
        return str(e)

def process_audio(audio_file):
    try:
        print("Processing audio file:", os.path.abspath(audio_file))
        if not os.path.exists(audio_file):
            return "Audio file does not exist."
        audio = AudioSegment.from_file(audio_file)
        duration_ms = 3 * 60 * 1000
        audio = audio[:duration_ms]

        temp_file = "temp_audio.wav"
        audio.export(temp_file, format="wav")

        audio = whisper_timestamped.load_audio(temp_file)
        audio = audio / np.max(np.abs(audio))
        model = whisper_timestamped.load_model("base", device="cpu")
        result = whisper_timestamped.transcribe(model, audio, vad = True, language="en")

        # Save results to SRT file
        writer = whisper_timestamped.utils.get_writer("srt", ".")
        writer(result, "output")
        print(f"SRT file saved: output.srt")

        # Save results to JSON file
        json_output_file = "output.json"
        with open(json_output_file, 'w', encoding='utf-8') as json_file:
            json.dump(result, json_file, indent=2, ensure_ascii=False)
        print(f"JSON file saved: {json_output_file}")

        return "Transcription completed"

    except Exception as e:
        return str(e)
    

def split_long_segments(segments, max_words=8, gap_threshold=1.0):
    refined_captions = []
    for caption in segments:
        words = caption['text'].split()
        start_time = caption['start']
        end_time = caption['end']
        duration = end_time - start_time

        phrases = re.split(r'([.,!?])', caption['text'])
        phrases = [''.join(x).strip() for x in zip(phrases[0::2], phrases[1::2])] + [phrases[-1]] if len(phrases) > 1 else [phrases[0]]
        
        phrase_start = start_time
        word_time = duration / len(words) if len(words) > 0 else 0

        for phrase in phrases:
            phrase_words = phrase.split()
            phrase_duration = word_time * len(phrase_words)
            phrase_end = phrase_start + phrase_duration

            # Track silences (pauses) by checking the gap between the start of the next phrase
            gap = phrase_start - refined_captions[-1]['end'] if refined_captions else 0

            # If the gap is larger than the threshold, split here
            if gap > gap_threshold:
                refined_captions.append({
                    'start': phrase_start,
                    'end': phrase_end,
                    'sentence': phrase
                })
                phrase_start = phrase_end
                continue

            # Break phrase into smaller chunks based on max_words
            for i in range(0, len(phrase_words), max_words):
                sub_phrase_words = phrase_words[i:i + max_words]
                sub_phrase_start = phrase_start + i * word_time
                sub_phrase_end = min(phrase_start + (i + max_words) * word_time, phrase_end)

                refined_captions.append({
                    'start': sub_phrase_start,
                    'end': sub_phrase_end,
                    'sentence': ' '.join(sub_phrase_words)
                })
            phrase_start = phrase_end
    return refined_captions

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"

def generate_srt(data):
    srt_lines = []
    for index, item in enumerate(data):
        start_time = format_time(item['start'])
        end_time = format_time(item['end'])
        sentence = item['sentence']

        srt_lines.append(f"{index + 1}")
        srt_lines.append(f"{start_time} --> {end_time}")
        srt_lines.append(sentence)
        srt_lines.append("")
    return "\n".join(srt_lines)

def save_srt(filename, srt_content):
    with open(filename, "w", encoding="utf-8") as file:
        file.write(srt_content)

# Function to remove specific files
def cleanup_files(file_paths):
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Removed: {file_path}")
        else:
            print(f"File not found: {file_path}")


# Streamlit app
st.title("YouTube/Video Transcription with Subtitles")
st.sidebar.title("Options")

# Input options
url = st.sidebar.text_input("YouTube URL")
uploaded_file = st.sidebar.file_uploader("Upload Video (.mp4)", type=["mp4"])

if st.sidebar.button("Start Transcription"):
    # List of files to remove at the beginning
    files_to_cleanup = ['downloaded_video.mp4', 'uploaded_video.mp4', 'corrected_output.srt', 'captioned_video.mp4', 'output.json','output.srt', "temp_audio.wav","extracted_audio.wav"]
    
    # Clean up the existing files
    cleanup_files(files_to_cleanup)

    video_file_path = ""
    if url:
        st.write("Downloading video from YouTube...")
        video_file_path = download_video_from_youtube(url)
        audio_file = download_audio_from_youtube(url)
    elif uploaded_file:
        st.write("Extracting audio from uploaded video...")
        with open("uploaded_video.mp4", "wb") as f:
            f.write(uploaded_file.read())
        audio_file = extract_audio_from_video("uploaded_video.mp4")
        video_file_path = "uploaded_video.mp4"
    else:
        st.error("Please provide a YouTube URL or upload a video.")
        st.stop()

    if audio_file:
        st.write("Processing audio using Whisper...")
        #print(audio_file)
        result_whisper = process_audio(audio_file)

        # Specify the path to your JSON file
        file_path = 'output.json'

        # Open and read the JSON file
        with open(file_path, 'r') as file:
            result = json.load(file)
        print(result["text"])

        st.write("Refining captions for correct segmentation...")
        refined_segments = split_long_segments(result["segments"])
        srt_content = generate_srt(refined_segments)
        save_srt("corrected_output.srt", srt_content)

        # Show some of the SRT content
        st.write("Corrected SRT Output:")
        num_lines_to_show = 20  # Number of lines to display
        srt_lines = srt_content.split("\n")[:num_lines_to_show]
        st.code("\n".join(srt_lines), language="plaintext")

        st.write("Generating video with subtitles...")
        video_output = "captioned_video.mp4"
      
        ffmpeg_cmd = [
            "ffmpeg", 
            "-y",  # Overwrite output files without asking
            "-i", video_file_path, 
            "-vf", f"subtitles=corrected_output.srt:force_style='PrimaryColour=&H0000FF00&,Fontsize=30'", 
            "-c:v", "libx264", 
            "-c:a", "copy", 
            video_output
        ]

        subprocess.run(ffmpeg_cmd)

        st.write("Process completed!")
        st.download_button("Download SRT", data=srt_content, file_name="output.srt", mime="text/plain")
        st.video(video_output)
