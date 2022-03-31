
from bs4 import BeautifulSoup 

class Parser():
    def village_multi_house(self, html_doc):
        soup = BeautifulSoup(html_doc, 'html.parser')
        result = {}
        for link in soup.find_all('div', "list_complex_info"): 
            # print(link.get_text())
            feature = link.find("dl", "complex_feature")
            dt = [f.string for f in feature.find_all("dt")]
            dd = [f.string for f in feature.find_all("dd")]
            village = str(link.h3.string)
            result[village] = {'info':{}, "items":[]}
            for d1, d2 in zip(dt, dd):
                d1 = str(d1)
                result[village]['info'][d1] = d2

            info = link.find("div", "complex_summary_info")
            dt = [f.string for f in info.find_all("dt")]
            dd = [f.string for f in info.find_all("dd")]
            for d1, d2 in zip(dt, dd):
                result[village]['info'][d1] = str(d2)

        for item in soup.find_all("div", "item_inner"):
            spans = item.find_all("span")
            span_dict = {i['class'][0]:[] for i in spans}
            for i in spans:
                span_dict[str(i['class'][0])].append(i.text)
            result[village]['items'].append(span_dict)
        print(result)
        return result
