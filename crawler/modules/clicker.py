
from core import Core

class Clicker(Core):
    def __init__(self, type="google_chrome"):
        super().__init__(type)

    def recursive_click(self, button_list:list):
        pass

    def click(self, button):
        pass
 


if __name__ == "__main__":
    clicker = Clicker()