
import time 
import json 
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import pyautogui

class Driver:
    def __init__(self, type):
        import sys
        import os
        os.environ["PATH"] += ":"+os.path.join(os.getcwd(), "data")
        if type =="google_chrome":
            self.driver = self.launch_chrome_driver(None) 

        else:
            raise ValueError("Not implemented driver engine")
    
    def launch_chrome_driver(self, options=None):
        from selenium import webdriver
        chrome_options = webdriver.ChromeOptions()
        # chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument("--window-size=1000x1000")
        chrome_options.add_argument('--disable-dev-shm-usage')
        driver =webdriver.Chrome('chromedriver',chrome_options=chrome_options)
        return driver

class Macro():
    def __init__(self):
        self.mouse_position = pyautogui.position()
        self.gui_size = pyautogui.size()
        self.figures ={
            "purple_background": "purple_background.png"
        }
    def move_mouse(self, x=None,y=None, wait=1, duration_seconds=0.01):
        time.sleep(wait)
        pyautogui.moveTo(x, y, duration=duration_seconds)
        print("[INFO] mouse is to position ",x,y)

    def click_mouse(self, num=1, wait=1, click_interval_sec=1):
        time.sleep(wait)
        x,y = self.mouse_position
        pyautogui.click(x=x, y=y, clicks=num, interval=click_interval_sec, button='left')
        print("[INFO] mouse is clicked")

    def find_all_figure_positions(self, figure):
        figure = self.figures[figure]
        self.figure_positions = list(pyautogui.locateAllOnScreen(figure, confidence=0.4))
        for p in range(len(self.figure_positions)):
            left, top, width, height = self.figure_positions[p]
            self.figure_positions[p] = {"center_x": left + width//2 ,
                                        "center_y" :top + height//2,
                                        "left" : left,
                                        "top": top, 
                                        "width": width, 
                                        "height": height}
        return self.figure_positions


class Crawller(Driver):
    def __init__(self, type="google_chrome"):
        super().__init__(type)
        with open("data/links.json") as json_file:
            self.links = json.load(json_file)
        self.elements = {}
        
    def find(self, tag=None, 
                   name=None, 
                   class_name=None, 
                   id=None, 
                   save=True, 
                   multiple=False, 
                   find_from_history=None,
                   find_from_element=None,
                   wait=1.0
        ):
        time.sleep(wait)

        if tag is not None:
            by = By.TAG_NAME
            value = tag
        elif name is not None:
            by = By.NAME
            value = name
        elif class_name is not None:
            by = By.CLASS_NAME
            value = class_name
        elif id is not None:
            by = By.ID
            value = id        

        if find_from_history is not None:
            engine = self.elements[find_from_history]
        elif find_from_element is not None:
            engine = find_from_element
        else:
            engine = self.driver

        elements = engine.find_elements(by=by, value=value)
        if len(elements) == 0:
            raise ValueError("Element List is empty with", engine, value ) 
        if not multiple:
            elements = elements[0]
        if save:
            self.elements[value] = elements
        
        return elements


    def click(self, element, wait=0.3, max_try=10):
        count = 0
        clicked = False
        time.sleep(wait)
        while not clicked and count < max_try:
            try:
                element.click()
                clicked = True
            except:
                count +=1
                print("[Warning] Click Failed with Try: ",count)
    def keyboard_input(self, element, value):
        element.send_keys(value)
        return element

    def keyboard_typing(self, element, string):
        element.send(string)
        return element

    def get(self, addr):
        if addr in self.links:
            addr = self.links[addr]
        html = self.driver.get(addr)
        return html

    def switch_window(self):
        self.click()
        
    @staticmethod
    def switch_to_last_window_decorator(function, *de_args, **de_kwargs):
        def switch_work(self, *args, **kwargs):
            main = self.driver.window_handles[0]
            self.driver.switch_to.window(self.driver.window_handles[-1])
            function(self, *args, **kwargs)
            self.driver.close()
            self.driver.switch_to.window(main)
            return None
        return switch_work

if __name__ == "__main__":
    crawller = Crawller()
    macro = Macro()
    crawller.get("네이버-부동산-매물")
    e = crawller.find(class_name="filter_region_inner")
    hierarchy = crawller.find(class_name="type_complex", find_from_history="filter_region_inner", multiple=True)
    index = 0
    crawller.click(hierarchy[0])
    list_warp = crawller.find(class_name="area_list_wrap")
    list_complex = crawller.find(tag="li", find_from_element=list_warp, multiple=True)
    LENGTH = len(list_complex)

    for index in range(LENGTH):
        try:
            # ---enter
            e = crawller.find(class_name="filter_region_inner")
            hierarchy = crawller.find(class_name="type_complex", find_from_history="filter_region_inner", multiple=True)
            crawller.click(hierarchy[0],wait=0.5)
            list_warp = crawller.find(class_name="area_list_wrap")
            list_complex = crawller.find(class_name="complex_item", find_from_element=list_warp, multiple=True)
            button = crawller.find(tag="a", find_from_element=list_complex[index])
            crawller.click(button, wait=0.5)
            # ---close
            detail_panel = crawller.find(class_name="detail_panel")
            close = crawller.find(class_name="btn_close", find_from_element=detail_panel)
            crawller.click(close)
        except:
            print("ERROR at ", index)
    print("---DONE---")
    time.sleep(10000)