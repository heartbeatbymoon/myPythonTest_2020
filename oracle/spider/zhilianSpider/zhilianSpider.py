#encoding=utf-8

import re
from urllib import parse

from util import util


class zhilianSpider:

    # 页数初始值
    page = 1
    # 默认处理结果为boo
    boo = True

    def loadPage(self):
        '''
        爬取页面源码
        :return:
        '''
        print ("正在爬取第" + str(self.page) + "页数据")
        # 要爬取的带关键字和页数的地址
        url = "https://sou.zhaopin.com/jobs/searchresult.ashx?sm=0&isadv=0&"
        url += parse.urlencode({"kw": searchName}) + "&" + parse.urlencode({"jl": searchCity}) + "&p=" + str(self.page)
        # print (url)
        #爬取url地址源代码
        html = util.getHtml(url)
        # print (html)
        return self.dealPage(html)

    def dealPage(self, html):
        '''
        处理数据
        :param html: 爬取到的页面html源码
        :return:
        '''
        # 先处理第一层数据
        pattern = re.compile(
            #<a style="font-weight: bold" par="ssidkey=y&amp;ss=201&amp;ff=03&amp;sg=8225b96d189f46f3b3f29dcded5c4f8d&amp;so=5&amp;uid=612300169" href="" target="_blank"><b>java</b>高级工程师</a>
            r'(<a style="font-weight: bold" par="ssidkey=y&amp;ss=201&amp;ff=03&amp;sg=\w+&amp;so=\d+&amp;uid=\d+"\shref=")(.*?)("\starget="_blank">.*?</a>)',re.S)
        message_list = pattern.findall(html)
        # print message_list
        if (len(message_list) > 0):
            # 循环出路径，进行第二次循环
            for message in message_list:
                try:
                    # 每一条招聘信息详细页的链接地址
                    path = message[1]
                    # print path
                    # 爬取每一条招聘信息详细页源代码
                    htmlInner = util.getHtml(path)
                    # print htmlInner
                    #公司名
                    company = util.getColumn('<h2><a.*?>(.*?)<img.*?></a></h2>', htmlInner)
                    #职位月薪
                    salary = util.getColumn('<span>职位月薪：</span><strong>(.*?)&nbsp;', htmlInner)
                    #工作经验
                    experience = util.getColumn('<li><span>工作经验：</span><strong>(.*?)</strong></li>', htmlInner)
                    #工作地点
                    city = util.getColumn('<li><span>工作地点：</span><strong><a target="_blank" href=".*?">(.*?)</a>(.*?)</strong></li>', htmlInner, False)
                    city = city.group(1)+city.group(2)
                    #学历
                    education = util.getColumn('<li><span>最低学历：</span><strong>(.*?)</strong></li>', htmlInner)
                    #职位类别
                    work = util.getColumn('<li><span>职位类别：</span><strong><a target="_blank" href=".*?">(.*?)</a></strong></li>', htmlInner)
                    #福利
                    welfare = util.replaceStr(
                        util.getColumn('<div style="width:683px;" class="welfare-tab-box">(.*?)</div>', htmlInner))
                    #职位描述
                    message_all = util.getColumn('<div class="tab-inner-cont">(.*?)</div>', htmlInner)
                    message = re.sub(re.compile(r'<.*?>',re.S),'',message_all).replace("查看职位地图","").replace("\n","").replace("\r","")
                    #技术关键字
                    keyWord = util.listToLowerAndDistinct(re.compile(r'[a-zA-Z]+\d*[a-zA-Z]*').findall(message), "-").replace("-nbsp", "").replace("nbsp", "")

                    # 要写入文件的一行数据信息
                    info = "compy&^" + company + "\tpay&^" + salary + "\tworkExp&^" + experience + "\tkeyWord&^"+ keyWord+ "\tcom.oracle.url&^" \
                           + path + "\txueli&^" + education + "\tcity&^" + city + "\tmessage&^" + message + "\twork&^" + work + "\tfuli&^" + welfare

                except Exception as e:
                    # 如果该条数据异常，忽略该条数据，继续执行
                    # print e.message
                    continue
                else:
                    #如果该条数据没有异常，写入文件
                    util.writePage(r"_" + searchCity + "_" + searchName, info, blackListKeyWord)
                    # info = (str(uuid.uuid4()),company,salary,experience,keyWord,path,education,city,message,work,welfare)
                    # util.writeDB(info)
        else:
            # 爬到最后的时候，给提示，并退出循环询问
            print ("已爬取到最后一页，没有更多数据")
            self.boo = False
        return self.boo


    def spider(self):
        '''
        爬虫主调度器
        :return:
        '''
        while True:
            # 爬取数据
            result = self.loadPage()
            # print result
            if (result == False):
                print ("爬取结束！")
                break
            else:
                p = input("是否继续爬取下一页数据(y/n)")
                if (p.lower() == "n"):
                    print ("爬取结束！")
                    break
                elif(p.lower() == "y"):
                    self.page += 1
                else:
                    print("输入内容有误")
                    continue

if __name__ == '__main__':
    #声明全局变量--搜索的职位
    global searchName
    #声明全局变量--搜索的城市
    global searchCity
    #声明全局变量--黑名单关键字
    global blackListKeyWord

    #控制台输入搜索的职位
    searchName = input("请输入要搜索的职位：")
    #控制台输入搜索的城市
    searchCity = input("请输入要搜索的城市：")
    #控制台输入黑名单关键字
    blackListKeyWord = input("请输入需要过滤的关键字：")
    #如果输入的职位为空，给定默认职位
    if searchName is None or searchName == "":
        searchName = "java"
    #如果输入的城市为空，给定默认城市
    if searchCity is None or searchCity == "":
        searchCity = "北京"
    #如果输入的黑名单关键字为空，给定默认关键字
    if blackListKeyWord is None or blackListKeyWord == "":
        blackListKeyWord = "实训,岗前"

    #创建爬虫类的实例化对象，并开始调用爬虫的主调度器
    zhilian = zhilianSpider()
    zhilian.spider()