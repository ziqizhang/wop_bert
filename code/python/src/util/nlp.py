from wordsegment import segment
import re
from urllib.parse import urlparse

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