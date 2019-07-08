#encoding=utf-8
from urllib import request,parse
import re
class baiduTieba():
    def spiderScheduler(self,name,startPage,endPage):
        print("开始爬去网页数据")
        for page in range(startPage,endPage+1):

            url = "https://tieba.baidu.com/f?" + parse.urlencode({"kw": name})
            url+="&"+parse.urlencode({"pn":(page-1)*50})
           # print(url)
        html=self.getHtml(url)
        self.dealPage(html)

    def getHtml(self,url):
        header = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.80 Safari/537.36"
        }
        req = request.Request(url, headers=header)
        response = request.urlopen(req)
        html = response.read().decode("utf-8")
        return html

    def dealPage(self,html):
        #<a rel="noreferrer" href="/p/5982636933" title="昨晚做了个梦" target="_blank" class="j_th_tit ">昨晚做了个梦</a>
       partten=re.compile(r'<a rel="noreferrer" href="/p/\d+" title=".*?" target="_blank" class="j_th_tit ">(.*?)</a>',re.S)
       titleList=partten.findall(html)
       # for title in titleList:
       #      print(title)
       flag="\n"
       mess=flag.join(titleList)
       self.writePage("D://tieba.txt",mess)

    def writePage(self,fileName,message):
        with open(fileName,"a",encoding="utf-8") as file:
            file.writelines(message)



if __name__=='__main__':
    name=input("请输入贴吧名")
    startPage=int(input("请输入起始页："))
    endPage=int(input("请输入zhongzhiye："))

    tieba=baiduTieba()
    tieba.spiderScheduler(name,startPage,endPage)
