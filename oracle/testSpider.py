#encoding=utf-8

from urllib import request

if __name__ == '__main__':

    url="http://www.baidu.com"
    html=request.urlopen(url).read()
    print(request.urlopen("http://www.baidu.com").read().decode("utf-8"))

