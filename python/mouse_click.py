import time 
import pyautogui 

def main():
    print("Hello ")

    for i in range(100):
        # pyautogui.moveTo(100, 200)
        x,y= pyautogui.position()
        print("INFO", x,y)
        time.sleep(2)
        pyautogui.doubleClick()
        # pyautogui.click(x=x, y=y, clicks=2, interval=0.25, button='left')

main()