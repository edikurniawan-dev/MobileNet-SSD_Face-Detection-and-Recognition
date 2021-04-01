from pynput.keyboard import Key, Controller
import time

time.sleep(10)

keyboard = Controller()
keyboard.press(Key.cmd)
keyboard.press('d')
keyboard.release(Key.cmd)
keyboard.release('d')

time.sleep(5)

keyboard = Controller()
keyboard.press(Key.cmd)
keyboard.press('d')
keyboard.release(Key.cmd)
keyboard.release('d')