
from Driver import Driver
import time 
import json 
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

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
                   find_from_history=False,
                   history_key=None
        ):

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

        if find_from_history:
            engine = self.elements[history_key]
        else:
            engine = self.driver

        elements = engine.find_elements(by=by, value=value)
        if not multiple:
            elements = elements[0]
        if save:
            self.elements[class_name] = elements
        return elements


    def click(self, element):
        element.click()

    def keyboard_input(self, element, value):
        element.send_keys(value)
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
    crawller.get("네이버-부동산-매물")
    e = crawller.find(class_name="search_input")
    crawller.click(e)
    e = crawller.keyboard_input(e, "강남")
    e = crawller.keyboard_input(e, Keys.ENTER)
    e = crawller.find(class_name="map_panel")
    time.sleep(10000)