

DEFAULT_CONFIG = {
    "start_address": "네이버-부동산-매물",
    "검색": ["구로구"],
    "maximum" : 3
}

from selenium.webdriver.common.keys import Keys

def village_multi_house(crawller, macro, config=DEFAULT_CONFIG):
    print("[INFO] Search ALL List")
    result = {}
    if config is None:
        config = DEFAULT_CONFIG
    print(config)
    
    crawller.get(config.get("start_address"))
    # --> 검색
    for keyword in config.get("검색"):
        print("[INFO] Keyword : ", keyword)
        e = crawller.find(id="search_input")
        e = crawller.keyboard_input(e, keyword)
        e = crawller.keyboard_input(e, Keys.ENTER)
        # ---
        MAXIMUM = min(1000, config.get("maximum"))
        result[keyword] = []
        for index in range(MAXIMUM):
                # --> enter to the last hierarchy 
                e = crawller.find(class_name="filter_region_inner")
                hierarchy = crawller.find(class_name="type_complex", find_from_history="filter_region_inner", multiple=True, wait=0.5)
                crawller.click(hierarchy[0],wait=1.0)
                # --> get the 단지 list
                list_warp = crawller.find(class_name="area_list_wrap", wait=0.5)
                list_complex = crawller.find(class_name="complex_item", find_from_element=list_warp, multiple=True, wait=0.5)
                if index >= len(list_complex):
                    break
                # --> find the button and click 
                button = crawller.find(tag="a", find_from_element=list_complex[index])
                crawller.click(button, wait=0.3)
                # --> Crawl HTML!! 
                innerHTML = crawller.driver.page_source
                result[keyword].append(innerHTML)
                print("[INFO] index : ", index)
                # --> Close the popup
                detail_panel = crawller.find(class_name="detail_panel")
                close = crawller.find(class_name="btn_close", find_from_element=detail_panel)
                crawller.click(close, wait=0.3)
    print("---DONE---")
    return result


