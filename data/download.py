import yt_dlp
import os
import argparse

def download_video(url, output_path):
    """
    Downloads a video from YouTube or other sources using yt-dlp.
    """
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def main():
    parser = argparse.ArgumentParser(description="Fetch .mp4 videos from YouTube or other sources.")
    parser.add_argument("--url", type=str, help="The URL of the video to download.")
    parser.add_argument("--output", type=str, default="data/raw", help="Output directory.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        
    if args.url:
        download_video(args.url, args.output)
    else:
        print("Please provide a URL.")

if __name__ == "__main__":
    main()
