import requests
import json
import datetime
import csv

with open("tesla.json", "r") as file:
    data = json.load(file)
timestamps_Tesla = data["chart"]["result"][0]["timestamp"]
adjclose_date_Tesla = data["chart"]["result"][0]["indicators"]["adjclose"][0][
    "adjclose"
]
with open("Nvidia.json", "r") as file:
    data = json.load(file)
adjclose_date_Nvidia = data["chart"]["result"][0]["indicators"]["adjclose"][0][
    "adjclose"
]
with open("Microsoft.json", "r") as file:
    data = json.load(file)
adjclose_date_Microsoft = data["chart"]["result"][0]["indicators"]["adjclose"][0][
    "adjclose"
]
with open("Intel.json", "r") as file:
    data = json.load(file)
adjclose_date_Intel = data["chart"]["result"][0]["indicators"]["adjclose"][0][
    "adjclose"
]
with open("Apple.json", "r") as file:
    data = json.load(file)
adjclose_date_Apple = data["chart"]["result"][0]["indicators"]["adjclose"][0][
    "adjclose"
]
with open("Amazon.json", "r") as file:
    data = json.load(file)
adjclose_date_Amazon = data["chart"]["result"][0]["indicators"]["adjclose"][0][
    "adjclose"
]

date = []
adjclose_Tesla = []
adjclose_Nvidia = []
adjclose_Microsoft = []
adjclose_Intel = []
adjclose_Apple = []
adjclose_Amazon = []
for i in range(
    len(timestamps_Tesla)
):  # timesstamp1-1574173800 timestamop last 1731940200 len = 1258
    dt = datetime.datetime.fromtimestamp(timestamps_Tesla[i])
    date.append(str(dt)[:10])
    adjclose_Tesla.append(str(adjclose_date_Tesla[i]))
    adjclose_Nvidia.append(str(adjclose_date_Nvidia[i]))
    adjclose_Microsoft.append(str(adjclose_date_Microsoft[i]))
    adjclose_Intel.append(str(adjclose_date_Intel[i]))
    adjclose_Apple.append(str(adjclose_date_Apple[i]))
    adjclose_Amazon.append(str(adjclose_date_Amazon[i]))


with open("Adj Close.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(
        ["Date", " Tesla", "Nvidia", "Microsoft", "Intel", "Apple", "Amazon"]
    )
    for i in range(len(date)):

        writer.writerow(
            [
                date[i],
                adjclose_Tesla[i],
                adjclose_Nvidia[i],
                adjclose_Microsoft[i],
                adjclose_Intel[i],
                adjclose_Apple[i],
                adjclose_Amazon[i],
            ]
        )
# print(len(adjclose_Nvidia), len(adjclose_date_Tesla))
