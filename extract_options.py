import os

from PIL import Image

# Downloaded from the Diamond Casino Heist Cheat Sheet on reddit
screenshot = Image.open("anotherscreenshot.png")
screenshot = screenshot.resize((1920, 1080))

top_left = (472, 267)
option_length = 125
option_gap = 19

x, y = top_left

for i in range(8):
    cropped = screenshot.crop((x, y, x+option_length, y+option_length))
    cropped.show()
    if i % 2 == 0:
        x += option_length + option_gap
    else:
        x = top_left[0]
        y += option_length + option_gap
