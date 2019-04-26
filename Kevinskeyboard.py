import keyboard
import time

def TestKeyboard(keyInput):
   keyboard.press_and_release(keyInput)

test = "Hi my name is Vinh"

for char in test:
    TestKeyboard(char)
    
