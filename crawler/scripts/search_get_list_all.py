

DEFAULT_CONFIG = {
    "start_address": "네이버-부동산-매물",
    "검색": "구로구",
    "maximum" : 5
}


from selenium.webdriver.common.keys import Keys

def search_get_list_all(crawller, macro, config=DEFAULT_CONFIG):
    result = {}
    if config is None:
        config = DEFAULT_CONFIG
    crawller.get(config.get("start_address"))
    # --- 검색
    e = crawller.find(id="search_input")
    e = crawller.keyboard_input(e, config.get("검색"))
    e = crawller.keyboard_input(e, Keys.ENTER)
    # ---
    e = crawller.find(class_name="filter_region_inner")
    hierarchy = crawller.find(class_name="type_complex", find_from_history="filter_region_inner", multiple=True)
    index = 0
    crawller.click(hierarchy[0])
    list_warp = crawller.find(class_name="area_list_wrap")
    list_complex = crawller.find(tag="li", find_from_element=list_warp, multiple=True)
    LENGTH =  min(len(list_complex), config.get("maximum"))

    for index in range(LENGTH):
        try:
            # ---enter
            e = crawller.find(class_name="filter_region_inner")
            hierarchy = crawller.find(class_name="type_complex", find_from_history="filter_region_inner", multiple=True)
            crawller.click(hierarchy[0],wait=0.5)
            list_warp = crawller.find(class_name="area_list_wrap")
            list_complex = crawller.find(class_name="complex_item", find_from_element=list_warp, multiple=True)
            button = crawller.find(tag="a", find_from_element=list_complex[index])
            crawller.click(button, wait=1.0)
            # --- CRAWLLL! 
            innerHTML = crawller.driver.get_attribute('innerHTML')
            result.append(innerHTML)
            # ---close
            detail_panel = crawller.find(class_name="detail_panel")
            close = crawller.find(class_name="btn_close", find_from_element=detail_panel)
            crawller.click(close, wait=1.0 )
        except:
            print("ERROR at ", index)
    print("---DONE---")
    return result


