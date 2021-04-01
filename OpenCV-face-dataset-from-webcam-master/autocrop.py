import win32gui
import win32con
from pynput.keyboard import Key, Controller

def get_window_hwnd(title):
    for wnd in enum_windows():
        if title.lower() in win32gui.GetWindowText(wnd).lower():
            return wnd
    return 0

def enum_windows():
    def callback(wnd, data):
        windows.append(wnd)

    windows = []
    win32gui.EnumWindows(callback, None)
    return windows

window = get_window_hwnd("MobileNet-SSD")
# window = get_window_hwnd("paint")
print(window)
#
# # window = win32gui.FindWindow("Paint", None)
# if window:
#     # tup = win32gui.GetWindowPlacement(window)
#     # if tup[1] == win32con.SW_SHOWMAXIMIZED:
#     #     print("maximized")
#     # elif tup[1] == win32con.SW_SHOWMINIMIZED:
#     #     print("minimized")
#     # elif tup[1] == win32con.SW_SHOWNORMAL:
#     #     print("normal")
#
#     win32gui.BringWindowToTop(window)
#     win32gui.SetForegroundWindow(window)
#     win32gui.ShowWindow(window, win32con.SW_NORMAL)

tup = win32gui.GetWindowPlacement(window)
if tup[1] == win32con.SW_SHOWMAXIMIZED:
    print("minimized")
    keyboard = Controller()
    keyboard.press(Key.cmd)
    keyboard.press('d')
    keyboard.release(Key.cmd)
