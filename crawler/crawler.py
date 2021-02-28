import urllib.request


def get_historical_data_from_naver(symbol):


    url = "https://finance.naver.com/marketindex/exchangeDailyQuote.nhn?marketindexCd="+symbol+"&page=1"
    req = urllib.request.urlopen(url)  # url에 대한 연결요청
    res = req.read()  # 연결요청에 대한 응답
    print (res)

if __name__ == '__main__':
    get_historical_data_from_naver('FX_USDKRW')

