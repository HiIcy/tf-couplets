# coding:utf-8
# __user__ = hiicy redldw
# __time__ = 2019/8/29
# __file__ = parse
# __desc__ =
from pathlib import Path
import codecs
import glob
import json

# TODO:协程
def read_data_to_disk(poetryfile):
    # print('come in ')
    lock.acquire()
    name = 'song' if 'song' in poetryfile else 'tang'
    with codecs.open(poetryfile,encoding='utf-8') as f:
        contents = json.load(f)
    with open(r'F:\Resources\kdata\chinesepoetry\read_data\{}.txt'.format(name),'a',encoding='utf-8') as fi:
        for content in contents:

            paragraphs = content['paragraphs']
            paragraphss = []
            for paragraph in paragraphs:
                try:
                    p1,p2 = paragraph.split("，")
                    if len(p1) == len(p2[:-1]):
                        paragraphss.append(p1+","+p2[:-1])
                except ValueError:
                    continue
            paragraphs = '.'.join(paragraphss).strip()
            author = content['author']
            title = content['title']
            if paragraphs == "":
                continue
            fi.write(f'{title}:{author}:{paragraphs}\n')
    lock.release()

def remodeling(couplet_file_in,couplet_file_out):
    couplet_file_inout = f'F:\Resources\kdata\couplet\parse_data\couplets.txt'
    with codecs.open(couplet_file_in,'r','utf-8') as cin:
        with codecs.open(couplet_file_out,'r','utf-8') as cout:
            with codecs.open(couplet_file_inout,'w','utf-8') as inout:
                try:
                    for linein,lineout in zip(cin,cout):
                        inline = linein.strip("\n\t")
                        inline = "".join(inline).replace("，",",").replace(" ","")
                        outline = lineout.strip("\n\t")
                        outline = "".join(outline).replace("，",",").replace(" ","")
                        inout.write(f'{inline} == {outline}\n')
                except Exception as e:
                    print(e)
                    exit(0)



# REW: 进程池的锁，在进程池初始化里就传进去
# 这种方式将Lock对象变为了所有子进程的全局对象
def init(l):
    global lock
    lock = l


if __name__ == "__main__":
    from multiprocessing import Pool, Lock
    # lock = Lock()
    # pool = Pool(5,initializer=init,initargs=(lock,))
    # source = r"F:\Resources\kdata\chinesepoetry\chinesepoetry\poet*.json"
    # poetryfiles = glob.glob(source)
    # for file in poetryfiles:
    #     pool.apply_async(read_data_to_disk, args=(file,))
    # pool.close()
    # pool.join()
    infile=r'F:\Resources\kdata\couplet\origin_data\test_in.txt'
    outfile=r'F:\Resources\kdata\couplet\origin_data\test_out.txt'
    remodeling(infile,outfile)