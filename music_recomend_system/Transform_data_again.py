# encoding=utf-8

import json
import sys


def is_null(s):
    return len(s.split(","))

def parse_song_info(song_info):
    try:
        song_id, name, artist, popularity = song_info.split(":::")
        return ",".join([song_id, "1.0", "1300000"])
    except Exception as e:
        return ""


def parse_playlist_line(in_line):
    try:
        contents = in_line.strip().split("\t")
        name, tags, playlist_id, subscribed_count = contents[0].split("##")
        songs_info = map(lambda x: playlist_id + "," + parse_song_info(x), contents[1:])
        songs_info = filter(is_null, songs_info)
        return "\n".join(songs_info)
    except Exception as e:
        print(e)
        return False


def parse_file(in_file, out_file):
    out = open(out_file, "w",encoding="utf-8")
    for line in open(in_file,"rb"):
        result = parse_playlist_line(line)
        if (result):
            out.write(result.encode("utf-8").decode().strip() + b"\n".decode())
    out.close()

# 为什么运行不了？？？？
parse_file("G:\\datas\\recommend\\playlistdetail.result.txt","G:\\datas\\recommend\\163_music_suprise_format.txt")