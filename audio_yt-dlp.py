from yt_dlp import YoutubeDL
from dotenv import load_dotenv
import os

load_dotenv()

OPERA_COOKIES_PATH = os.getenv("OPERA_COOKIES_PATH")
AUDIO_OUTPUT_DIR = os.getenv("AUDIO_OUTPUT_DIR", "audio")
path_template = f"{AUDIO_OUTPUT_DIR}/%(title)s.%(ext)s"

def audio_yt_dlp(yt_link):
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": path_template,
        "cookiesfrombrowser": ('opera', OPERA_COOKIES_PATH),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }]
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([yt_link])

if __name__ == "__main__":
    link = "https://www.youtube.com/playlist?list=PLHtUOYOPwzJGGZkjR-FspIL17YtSBGaCR"
    audio_yt_dlp(link)