import os

from PIL import Image

# Downloaded from the Diamond Casino Heist Cheat Sheet on reddit
img = Image.open("diamondcasinoheist.png")

top_lefts = [(107, 773), (869, 773), (107, 1731), (869, 1731)]

fp_answers_directory = "fp_answers"

fingerprint_answer = 1
if not os.path.exists(fp_answers_directory):
    os.makedirs(fp_answers_directory)

for x, y in top_lefts:
    directory = f"f{fingerprint_answer}"
    if not os.path.exists(os.path.join(fp_answers_directory, directory)):
        os.makedirs(os.path.join(fp_answers_directory, directory))
    # iterate 4 times
    move_selection = [0, 134, 131, 132]
    for index, i in enumerate(move_selection, start=1):

        x += i

        cropped = img.crop((x, y, x+114, y+114))
        cropped.save(os.path.join(fp_answers_directory, directory) + f"/{index}.png")
    fingerprint_answer += 1
