import os
import sys
import time
import json
import asyncio
import requests
from bs4 import BeautifulSoup
from google import genai
from gtts import gTTS
import whisper
import warnings
warnings.filterwarnings("ignore")
from playwright.async_api import async_playwright
from moviepy.editor import ImageClip, AudioFileClip, TextClip, CompositeVideoClip, concatenate_videoclips

from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials

# --- SECRETS & CONFIG ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
YOUTUBE_CLIENT_ID = os.environ.get("YOUTUBE_CLIENT_ID")
YOUTUBE_CLIENT_SECRET = os.environ.get("YOUTUBE_CLIENT_SECRET")
YOUTUBE_REFRESH_TOKEN = os.environ.get("YOUTUBE_REFRESH_TOKEN")

VOICE = "en-US-ChristopherNeural"
OUTPUT_AUDIO = "voiceover.mp3"
OUTPUT_SRT = "subtitles.srt"
OUTPUT_VIDEO = "final_reel.mp4"
REPOS_FILE = "repos.json"
SCRIPT_FILE = "script.txt"

def fetch_repos():
    print("[1] Fetching trending GitHub repos...")
    resp = requests.get("https://github.com/trending/python?since=daily")
    soup = BeautifulSoup(resp.text, "html.parser")
    repos = []
    
    for article in soup.find_all('article', class_='Box-row')[:3]:
        repo_name = article.find('h2', class_='h3').text.strip().replace('\n', '').replace(' ', '')
        desc_p = article.find('p', class_='col-9')
        repos.append({
            "name": repo_name, 
            "url": f"https://github.com/{repo_name}", 
            "description": desc_p.text.strip() if desc_p else "Trending repository."
        })
    
    # Save for the next workflow step
    with open(REPOS_FILE, "w") as f:
        json.dump(repos, f)
    print("  Saved trending repos to", REPOS_FILE)

def generate_script():
    print("[2] Generating script...")
    with open(REPOS_FILE, "r") as f:
        repos = json.load(f)
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = "Write a fast-paced 45-second voiceover script for an Instagram Reel highlighting these 3 GitHub repos. No visual directions, just the spoken words. Give a quick intro, mention each repo and what it does, and a quick outro:\n" + str(repos)
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model='gemini-3.5-flash-lite', contents=prompt)
            script_text = response.text.replace('*', '').replace('#', '').strip()
            # Save for the next workflow step
            with open(SCRIPT_FILE, "w") as f:
                f.write(script_text)
            print("  Script generated and saved.")
            break
        except Exception as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                wait_time = 15 * (attempt + 1)
                print(f"Gemini API busy (503). Retrying in {wait_time}s...")
                time.sleep(wait_time)
                if attempt == max_retries - 1:
                    raise Exception("Gemini API failed after 5 attempts.")
            else:
                raise e

def generate_audio():
    print("[3] Synthesizing Voice & Subtitles...")
    with open(SCRIPT_FILE, "r") as f:
        script_text = f.read()
    tts = gTTS(text=script_text, lang='en', tld='com')
    tts.save(OUTPUT_AUDIO)
    print("    Transcribing audio to generate exact subtitle timings...")
    model = whisper.load_model("base")
    result = model.transcribe(OUTPUT_AUDIO, word_timestamps=True)
    with open(OUTPUT_SRT, "w", encoding="utf-8") as srt_file:
        srt_index = 1
        for segment in result['segments']:
            for word in segment.get('words', []):
                text = word['word'].strip()
                if not text: continue
                
                def fmt_time(s):
                    h, m, sec, ms = int(s // 3600), int((s % 3600) // 60), int(s % 60), int((s % 1) * 1000)
                    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"
                srt_file.write(f"{srt_index}\n{fmt_time(word['start'])} --> {fmt_time(word['end'])}\n{text}\n\n")
                srt_index += 1
    print("  Voice and Subtitles generated.")

async def capture_visuals():
    print("[4] Capturing Screenshots...")
    with open(REPOS_FILE, "r") as f:
        repos = json.load(f)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={'width': 1080, 'height': 1920})
        for i, repo in enumerate(repos):
            # Changed wait_until to "load" and added a 60s timeout
            await page.goto(repo["url"], wait_until="load", timeout=60000)
            
            # Added a 2-second wait to allow dynamic elements/fonts to load
            await page.wait_for_timeout(2000)
            
            path = f"repo_{i}.png"
            await page.screenshot(path=path)
        await browser.close()
    print("  Screenshots captured.")

def assemble_video():
    print("[5] Rendering Video...")
    with open(REPOS_FILE, "r") as f:
        repos = json.load(f)
        
    image_paths = [f"repo_{i}.png" for i in range(len(repos))]
    audio = AudioFileClip(OUTPUT_AUDIO)
    time_per_image = audio.duration / len(image_paths)
    
    bg_clips = [ImageClip(img).set_duration(time_per_image) for img in image_paths]
    base_video = concatenate_videoclips(bg_clips, method="compose").set_audio(audio)
    
    with open(OUTPUT_SRT, 'r', encoding='utf-8') as f:
        blocks = f.read().strip().split('\n\n')
    
    subtitle_clips = []
    for block in blocks:
        lines = block.split('\n')
        if len(lines) >= 3:
            start_str, end_str = lines[1].split(' --> ')
            def to_sec(t_str):
                h, m, s_ms = t_str.split(':')
                return int(h)*3600 + int(m)*60 + int(s_ms.split(',')[0]) + int(s_ms.split(',')[1])/1000.0
            
            txt_clip = TextClip(" ".join(lines[2:]), fontsize=90, color='white', font='Liberation-Sans-Bold', bg_color='rgba(0,0,0,0.6)')
            txt_clip = txt_clip.set_start(to_sec(start_str)).set_end(to_sec(end_str)).set_position(('center', 'center'))
            subtitle_clips.append(txt_clip)
            
    final_video = CompositeVideoClip([base_video] + subtitle_clips)
    
    # 1. THE FIX: Force the entire composition to match the audio track's exact length.
    # This prevents Whisper subtitles from artificially stretching the video stream.
    final_video = final_video.set_duration(audio.duration)
    # 2. Added audio_bitrate="128k" to ensure the standard format
    final_video.write_videofile(
        OUTPUT_VIDEO, 
        fps=30, 
        codec="libx264", 
        audio_codec="aac", 
        audio_fps=44100, 
        audio_bitrate="128k",
        ffmpeg_params=["-pix_fmt", "yuv420p", "-profile:v", "main"]
    )
    print("  Video rendered.")

def publish_video():
    print("[6] Publishing to YouTube Shorts...")
    
    if not all([YOUTUBE_CLIENT_ID, YOUTUBE_CLIENT_SECRET, YOUTUBE_REFRESH_TOKEN]):
        print("Skipping publish: Missing YouTube secrets.")
        return

    # Reconstruct credentials using the permanent refresh token
    credentials = Credentials(
        token=None, 
        refresh_token=YOUTUBE_REFRESH_TOKEN,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=YOUTUBE_CLIENT_ID,
        client_secret=YOUTUBE_CLIENT_SECRET
    )

    youtube = build("youtube", "v3", credentials=credentials)

    print("    Uploading video directly to YouTube...")
    
    # Adding "#shorts" to the title/description and ensuring the video is 
    # under 60 seconds (vertical) automatically categorizes it as a Short.
    request_body = {
        "snippet": {
            "title": "Trending GitHub Repos! #shorts #coding #github",
            "description": "Check out these trending Python repositories on GitHub today! #coding #python #developer",
            "tags": ["shorts", "coding", "github", "programming", "python", "developer"],
            "categoryId": "28" # 28 = Science & Technology
        },
        "status": {
            "privacyStatus": "public", # Use "private" or "unlisted" for testing
            "selfDeclaredMadeForKids": False
        }
    }

    # Chunksize=-1 uploads the file in a single request (fine for <60s Reels)
    media_file = MediaFileUpload(OUTPUT_VIDEO, chunksize=-1, resumable=True)

    try:
        request = youtube.videos().insert(
            part="snippet,status",
            body=request_body,
            media_body=media_file
        )
        response = request.execute()
        
        print(f"  Published! Video ID: {response['id']}")
        print(f"  URL: https://youtube.com/shorts/{response['id']}")
    except Exception as e:
        raise Exception(f"YouTube upload failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a step to execute (e.g., python main.py fetch)")
        sys.exit(1)
        
    step = sys.argv[1]
    
    if step == "fetch": fetch_repos()
    elif step == "script": generate_script()
    elif step == "audio": generate_audio()
    elif step == "visuals": asyncio.run(capture_visuals())
    elif step == "video": assemble_video()
    elif step == "publish": publish_video()
    else: print(f"Unknown step: {step}")
