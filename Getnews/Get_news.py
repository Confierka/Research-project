import requests
import json

headers = {
    "x-rapidapi-key": "568e51a4f8msh5728861c8085fc4p1fa79fjsnbb70df0840eb",
    "x-rapidapi-host": "seeking-alpha.p.rapidapi.com",
}
querystring = {
    "until": "1731965751",
    "since": "1574112951",
    "size": "40",
    "number": "",
    "id": "msft",
}

BASE_LINK = "https://seekingalpha.com"

url = "https://seeking-alpha.p.rapidapi.com/news/v2/list-by-symbol"
res = dict()


def req(page):
    response = requests.get(url, headers=headers, params=querystring).json()
    for new in response["data"]:
        buff = {
            new["id"]: {
                "title": new["attributes"]["title"],
                "url": BASE_LINK + new["links"]["self"],
                "time": new["attributes"]["publishOn"],
            }
        }
        res.update(buff)
    print(page)


for page in range(1, 80):
    querystring["number"] = page

    try:
        req(page)
    except:
        req(page)


with open("INTC_news.json", "w") as f:
    json.dump(res, f, ensure_ascii=False, indent=4)
