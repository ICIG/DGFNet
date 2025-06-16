import json
import os
import subprocess

def download(label, name, path):
    label = label.replace(" ", "_")
    
    path_data = os.path.join(path, label)
    if not os.path.exists(path_data):
        os.makedirs(path_data)

    link_prefix = "https://www.youtube.com/watch?v="
    filename = os.path.join(path_data, name) + ".mp4"
    link = link_prefix + name

    if os.path.exists(filename):
        print(f"已存在，跳过: [{label}] - [{name}]")
        return

    print(f"正在下载视频: [{label}] - [{name}]")
    command = [
        'yt-dlp', link,
        '-o', filename,
        #'-f', 'best',
        '-q'  # 抑制日志
    ]

    try:
        subprocess.run(command, check=True)
        print(f'下载完成: {filename}')
    except subprocess.CalledProcessError as e:
        print(f"下载视频出错 [{label}] - [{name}]: {e}")
    return


music_dat = 'D:\研究\代码\iQuery-main\data\json\MUSIC21_solo_videos.json'
video_pth = '../MUSIC21_dataset/videos/solo'

try:
    with open(music_dat, "r") as read_file:
        data = json.load(read_file)
except FileNotFoundError:
    print(f"错误: 找不到 JSON 文件 '{music_dat}'。")
    exit(1)
except json.JSONDecodeError:
    print(f"错误: 解析 JSON 文件 '{music_dat}' 失败。")
    exit(1)

if 'videos' in data:
    for music in data['videos']:
        v = data['videos'][music]
        for vid_name in v:
            download(music, vid_name, video_pth)
else:
    print("错误: JSON 格式不正确，找不到 'videos' 键。")