# encoding=utf-8

import json
import sys

def parse_song_line(in_line):
    # json.loads()函数是将json格式数据转换为字典
    data = json.loads(in_line)
    name = data["result"]["name"]
    tags = ",".join(data["result"]["tags"])
    subscribed_count = data["result"]["subscribedCount"]
    if(subscribed_count<100):
        return False
    playlist_id = data["result"]["id"]
    song_info = ""
    songs = data["result"]["tracks"]
    for song in songs:
        try:
            song_info += "\t"+":::".join([str(song["id"]),song["name"],song["artists"][0]["name"],str(song["popularity"])])
        except Exception as e:
            continue
    return name+"##"+tags+"##"+str(playlist_id)+"##"+str(subscribed_count)+song_info


def parse_file(in_file, out_file):
    out = open(out_file, "w",encoding="utf-8")
    for line in open(in_file,"rb"):
        result = parse_song_line(line)
        if (result):
            out.write(result.encode("utf-8").strip().decode() + b"\n".decode())
    out.close()


# .json文件是jbk格式的？
parse_file("G:\\datas\\recommend\\playlistdetail.all.json","G:\\datas\\recommend\\playlistdetail.result.txt")
