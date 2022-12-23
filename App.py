from tkinter import *
from tkinter import ttk
from PIL import ImageTk, Image
from tkinter import filedialog
import os

import subprocess   

path_img1 = ""
path_img2 = ""


def openfn():
    filename = filedialog.askopenfilename(initialdir = "./data/",
                                           title = "Select a File",
                                           filetypes = (("Images files",
                                                         "*.jpg*"),
                                                        ("all files",
                                                         "*.*")))
    return filename


def open_img(i):
    x = openfn()
    img = Image.open(x)
    img = img.resize((250, 250), Image.Resampling.LANCZOS)
    img2 = ImageTk.PhotoImage(img)
    panel = Label(window, image=img2)
    panel.image = img2
    panel.grid(column = i, row = 6)
    if i == 1:
        global path_img1
        path_img1 = x
        btn_img1.configure(text="Image 1")
    elif i == 2:
        global path_img2
        path_img2 = x
        btn_img2.configure(text="Image 2")


window = Tk()
window.title('Application Autour des visages')
window.config(background = "white")
window.resizable(width=False, height=False)
window.minsize(width=500, height=200)

label_gen = Label(window,
                             text = "Partie Generation",
                             width = 100, height = 2,
                            )

label_interpole = Label(window,
                             text = "Partie Interpolation",
                             width = 100, height = 2,
                            ) 
label_separator = Label(window,
                             width = 100, height = 2,
                            ) 
#Button to open image 1
btn_img1 = Button(window, text="Ouvrir l'image 1", command= lambda:open_img(1))
#Button to open image 2
btn_img2 = Button(window, text="Ouvrir l'image 2", command=lambda:open_img(2))
#Button to Generate with AE
btn_genere_AE = Button(window, text='Generation_AE', command=lambda:[subprocess.run("python ./visage/Generation_AE.py", shell = True)])
#Button to Generate with VAE
btn_genere_VAE = Button(window, text='Generation_VAE', command=lambda:[subprocess.run("python ./visage/Generation_VAE.py", shell = True)])

#Button to interpolate
btn_interpole = Button(window, text='Interpoler', command=lambda:[subprocess.run("python ./visage/Interpolation_AE_VAE.py" + " " + path_img1 + " " + path_img2 , shell = True)])

#Button to exit/reset
button_reset = Button(window, text = "Restart", command = lambda:[window.destroy(), os.system('python App.py')])
button_exit = Button(window, text = "Exit", command = exit)

#partie generation
label_gen.grid(column = 1, row = 1, columnspan = 2)    
btn_genere_AE.grid(column = 1, row = 3)
btn_genere_VAE.grid(column = 2, row = 3)

#partie interpolation
label_interpole.grid(column = 1, row = 4, columnspan = 2) 
btn_img1.grid(column = 1, row = 5)
btn_img2.grid(column = 2, row = 5)

btn_interpole.grid(column = 1, row = 7, columnspan = 2)

#separator
ttk.Separator(
    master=window,
    orient=HORIZONTAL,
    style='blue.TSeparator',
    class_= ttk.Separator,
    takefocus= 1,
    cursor='plus'    
).grid(column=1, row = 9, columnspan = 4, ipadx=200, pady=10)

button_reset.grid(column = 1, row = 10)
button_exit.grid(column = 2, row = 10)


window.mainloop()

