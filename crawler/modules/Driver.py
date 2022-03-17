class Driver:
    def __init__(self, type):
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