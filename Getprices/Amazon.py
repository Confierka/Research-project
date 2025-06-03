import requests
import json


cookies = {
    "tbla_id": "78da8c7a-c265-4cff-b8d3-88771ffc4e1a-tuctb139752",
    "gpp": "DBAA",
    "gpp_sid": "-1",
    "cmp": "t=1732009409&j=0&u=1---",
    "A3": "d=AQABBJK9GWQCEOUpvmyyK9DUqiGbhjjiAXQFEgABCAGvPWdlZ-2Nb2UB9qMAAAcIkr0ZZDjiAXQ&S=AQAAAv00iR4uqycbiG0M1qrhIX8",
    "EuConsent": "CQIVlIAQIVlIAAOACBNLBQFoAP_gAEPgACiQKZNB9G7WTXFneXp2YPskOYUX0VBJ4MAwBgCBAcABzBIUIBwGVmAzJEyIICACGAIAIGBBIABtGAhAQEAAYIAFAABIAEgAIBAAIGAAACAAAABACAAAAAAAAAAQgEAXMBQgmAZEBFoIQUhAggAgAQAAAAAEAIgBCgQAEAAAQAAICAAIACgAAgAAAAAAAAAEAFAIEQAAAAECAotkfQAAAAAAAAAAAAAAAAABBTIAEg1KiAIsCQkIBAwggQAiCgIAKBAEAAAQIAAACYIChAGACowEQAgBAAAAAAAAAAQAIAAAIAEIAAgACBAAAAABAAEABAIAAAQAAAAAAAAAAAAAAAAAAAAAAAAAxACEEAAIAIIACCgAAAAEAAAAAAAAABEAAQAAAAAAAAAAAAABEAAAAAAAAAAAAAAAAAABAAAAAAAAAEAIgsAAAAAAAAAAAAAAAAAAIAA",
    "GUC": "AQABCAFnPa9nZUIXyQPW&s=AQAAANzimCNw&g=ZzxfEg",
    "A1": "d=AQABBJK9GWQCEOUpvmyyK9DUqiGbhjjiAXQFEgABCAGvPWdlZ-2Nb2UB9qMAAAcIkr0ZZDjiAXQ&S=AQAAAv00iR4uqycbiG0M1qrhIX8",
    "axids": "gam=y-vRvXo6dE2uJnbvjQk083Bs4U9sbGdC_D~A&dv360=eS1tTzlkSEE1RTJ1SFZRb0RqSW5RcnJwVGJSUkhlRXJobH5B&ydsp=y-9a1s7EZE2uJmYWXPjSqajDRSndoQ6cK7~A&tbla=y-DS3Kee9E2uIzHP3CWCIvMshEjzspENja~A",
    "A1S": "d=AQABBJK9GWQCEOUpvmyyK9DUqiGbhjjiAXQFEgABCAGvPWdlZ-2Nb2UB9qMAAAcIkr0ZZDjiAXQ&S=AQAAAv00iR4uqycbiG0M1qrhIX8",
    "PRF": "t%3DAMZN%252BTSLA%252BNVDA%252BINTC%252BAAPL%252BMSFT%252BDIS%252BMCD%252B%255ESP500V%252B%255EGSPC%252BES%253DF%252BBA%252BRBLX%252BDJT%252BFIVE%26qke-neo%3Dfalse",
}

headers = {
    "accept": "*/*",
    "accept-language": "ru,ru-RU;q=0.9,be-BY;q=0.8,be;q=0.7,en-US;q=0.6,en;q=0.5",
    # Requests sorts cookies= alphabetically
    # 'cookie': 'tbla_id=78da8c7a-c265-4cff-b8d3-88771ffc4e1a-tuctb139752; gpp=DBAA; gpp_sid=-1; cmp=t=1732009409&j=0&u=1---; A3=d=AQABBJK9GWQCEOUpvmyyK9DUqiGbhjjiAXQFEgABCAGvPWdlZ-2Nb2UB9qMAAAcIkr0ZZDjiAXQ&S=AQAAAv00iR4uqycbiG0M1qrhIX8; EuConsent=CQIVlIAQIVlIAAOACBNLBQFoAP_gAEPgACiQKZNB9G7WTXFneXp2YPskOYUX0VBJ4MAwBgCBAcABzBIUIBwGVmAzJEyIICACGAIAIGBBIABtGAhAQEAAYIAFAABIAEgAIBAAIGAAACAAAABACAAAAAAAAAAQgEAXMBQgmAZEBFoIQUhAggAgAQAAAAAEAIgBCgQAEAAAQAAICAAIACgAAgAAAAAAAAAEAFAIEQAAAAECAotkfQAAAAAAAAAAAAAAAAABBTIAEg1KiAIsCQkIBAwggQAiCgIAKBAEAAAQIAAACYIChAGACowEQAgBAAAAAAAAAAQAIAAAIAEIAAgACBAAAAABAAEABAIAAAQAAAAAAAAAAAAAAAAAAAAAAAAAxACEEAAIAIIACCgAAAAEAAAAAAAAABEAAQAAAAAAAAAAAAABEAAAAAAAAAAAAAAAAAABAAAAAAAAAEAIgsAAAAAAAAAAAAAAAAAAIAA; GUC=AQABCAFnPa9nZUIXyQPW&s=AQAAANzimCNw&g=ZzxfEg; A1=d=AQABBJK9GWQCEOUpvmyyK9DUqiGbhjjiAXQFEgABCAGvPWdlZ-2Nb2UB9qMAAAcIkr0ZZDjiAXQ&S=AQAAAv00iR4uqycbiG0M1qrhIX8; axids=gam=y-vRvXo6dE2uJnbvjQk083Bs4U9sbGdC_D~A&dv360=eS1tTzlkSEE1RTJ1SFZRb0RqSW5RcnJwVGJSUkhlRXJobH5B&ydsp=y-9a1s7EZE2uJmYWXPjSqajDRSndoQ6cK7~A&tbla=y-DS3Kee9E2uIzHP3CWCIvMshEjzspENja~A; A1S=d=AQABBJK9GWQCEOUpvmyyK9DUqiGbhjjiAXQFEgABCAGvPWdlZ-2Nb2UB9qMAAAcIkr0ZZDjiAXQ&S=AQAAAv00iR4uqycbiG0M1qrhIX8; PRF=t%3DAMZN%252BTSLA%252BNVDA%252BINTC%252BAAPL%252BMSFT%252BDIS%252BMCD%252B%255ESP500V%252B%255EGSPC%252BES%253DF%252BBA%252BRBLX%252BDJT%252BFIVE%26qke-neo%3Dfalse',
    "origin": "https://finance.yahoo.com",
    "priority": "u=1, i",
    "referer": "https://finance.yahoo.com/quote/AMZN/history/?period1=1574164892&period2=1732017681",
    "sec-ch-ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-site",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
}

params = {
    "events": "capitalGain|div|split",
    "formatted": "true",
    "includeAdjustedClose": "true",
    "interval": "1d",
    "period1": "1574164892",
    "period2": "1732017681",
    "symbol": "AMZN",
    "userYfid": "true",
    "lang": "en-US",
    "region": "US",
}

response = requests.get(
    "https://query1.finance.yahoo.com/v8/finance/chart/AMZN",
    params=params,
    cookies=cookies,
    headers=headers,
)

res = json.loads(response.text)

with open("Amazon.json", "w") as file:
    file.write(json.dumps(res))
