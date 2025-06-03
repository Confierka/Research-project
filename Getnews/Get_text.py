import requests
import json

url = "https://seeking-alpha.p.rapidapi.com/news/get-details"

headers = {
    "x-rapidapi-key": "568e51a4f8msh5728861c8085fc4p1fa79fjsnbb70df0840eb",
    "x-rapidapi-host": "seeking-alpha.p.rapidapi.com",
}

with open("INTC_news.json", "r") as f:
    file_data = json.load(f)

a = 1638
dict_news = list()
bad_news = list()


def main():
    for i, id in enumerate(list(file_data.keys())[a:]):
        querystring = {"id": id}
        try:
            response = requests.get(url, headers=headers, params=querystring)
            if response.status_code != 200 and response.status_code != 302:
                break
            elif response.status_code == 302:
                bad_news.append(list(file_data.keys())[i])
            else:
                dict_news.append(response.json())
                print(i)
        except:
            break

    with open(f"alpha_api{a}.json", "w") as f:
        json.dump(dict_news, f, ensure_ascii=False, indent=4)
    with open(f"Bad_news{a}.json", "w") as f:
        json.dump(bad_news, f, ensure_ascii=False, indent=4)


main()
