from wordsegment import segment
import re
from urllib.parse import urlparse

url_stopwords=["html","htm","aspx","asp","jsp","cgi"]


def url_to_words(url:str):
    try:
        o = urlparse(url)
        host = o.hostname
        if host is None:
            host=""
        else:
            host=host.split(".")
            index=-1
            length=-1
            for i in range(len(host)):
                p = host[i]
                if len(p)>length:
                    length=len(p)
                    index=i
            if index>-1 and len(host[index])>3:
                host=host[index]
                host=" ".join(segment(host))

            else:
                host=""

        path= re.sub('[^0-9a-zA-Z]+', ' ', o.path).strip()

        return host+" "+path

    except:
        return ""


'''
option: 0 - keep host only if it's multi-word, path (no query/params). removing stopwords, single letters
1 - 0 but no host
2 - 0 but only host
'''
def url_to_words_basic(url:str, option=0):
    try:
        o = urlparse(url)
        host = o.hostname
        if host is None:
            host=""
        else:
            host=host.split(".")
            index=-1
            length=-1
            for i in range(len(host)):
                p = host[i]
                if len(p)>length:
                    length=len(p)
                    index=i
            if index>-1 and len(host[index])>3:
                host=host[index]
                host=" ".join(segment(host))

            else:
                host=""

        path= re.sub('[^0-9a-zA-Z]+', ' ', o.path).strip()

        if option==0:
            if len(host.split(" "))>1:
                concat= host+" "+path
                return clean_url_words(concat)
            else:
                return clean_url_words(path)
        elif option==1:
            return clean_url_words(path)
        elif option==2:
            return clean_url_words(host)

    except:
        return ""

def clean_url_words(url_as_words:str):
    toks = url_as_words.split(" ")
    new_list = [x for x in toks if (x not in url_stopwords)]
    concat = ""
    for t in new_list:
        if len(t) < 3:
            continue
        concat += t + " "
    return concat.strip()