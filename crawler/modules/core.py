

class Core:
    def __init__(self, type):
        if type =="google_chrome":
            self.driver = self.launch_chrome_driver(None) 
        else:
            raise ValueError("Not implemented driver engine")

        

    def launch_chrome_driver(self, options):
        from selenium import webdriver
        driver =webdriver.Chrome('chromedriver',chrome_options=options) 
        return driver