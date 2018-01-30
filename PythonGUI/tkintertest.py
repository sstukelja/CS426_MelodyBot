import tkinter as tk
from tkinter import filedialog
from tkinter import Menu
from tkinter import Scale
from tkinter import Frame
from tkinter import IntVar
from tkinter import Entry
from tkinter import Radiobutton
import pygame

#Import functions
def get_file_name(x):
    start = x.rfind('/') + 1
    return x[start:len(x)]

def add_song():
    file_path = filedialog.askopenfilename()
    song_tuple = (get_file_name(file_path), file_path)
    list_songs.append(song_tuple)
    listbox_songs.insert(tk.END, song_tuple[0])

# Playback functions
def play_song():
    pygame.mixer.music.load((list_songs[(listbox_songs.curselection()[0])])[1])
    pygame.mixer.music.play()
    button_play.grid_forget()
    button_stop.grid(row=1, column=0)

def pause_song():
    pygame.mixer.music.stop()
    button_stop.grid_forget()
    button_play.grid(row=1, column=0)

def set_time():
    slider_time.set(time_value.get())

# Initilization of window application
window = tk.Tk()

window.title("MelodyBot")
window.geometry("1200x600")
window.configure(background="#e2faff")

pygame.mixer.init()

# Frames
frame_options = Frame(master=window, width=300, height=500)
frame_options.grid(row=0, column=1)

frame_media = Frame(master=window, width=100, height=100, bg="white", padx=100)
frame_media.grid(row=0, column=0)

frame_radio = Frame(master=frame_options, width=200, height=50)
frame_radio.grid(row=1, column=0)

# List of tuples containing file information
list_songs = list()

# Listbox containing user-facing song names
listbox_songs = tk.Listbox(master=frame_media, width=50, selectmode=tk.BROWSE)
listbox_songs.grid(row=0, column=0)

# Menus
menu_bar = Menu(window)
window.configure(menu=menu_bar)

menu_file = Menu(menu_bar)
menu_edit = Menu(menu_bar)
menu_help = Menu(menu_bar)

menu_file.add_command(label="Import Song...",command=add_song)
menu_file.add_separator()
menu_file.add_command(label="Exit...")
menu_edit.add_command(label="Choose Settings...")
menu_help.add_command(label="About...")

menu_bar.add_cascade(label="File", menu=menu_file)
menu_bar.add_cascade(label="Edit", menu=menu_edit)
menu_bar.add_cascade(label="Help", menu=menu_help)

# Buttons controlling music playback

#Play button
load_image_play = tk.PhotoImage(file="C:/Users/Lepi/Desktop/CS425/PythonGUI/Play.png")
button_play = tk.Button(master=frame_media, image=load_image_play, command=play_song)
button_play["bg"] = "white"
button_play["border"] = "0"
button_play.grid(row=1, column=0)

#Stop button
load_image_stop = tk.PhotoImage(file="C:/Users/Lepi/Desktop/CS425/PythonGUI/Pause.png")
button_stop = tk.Button(master=frame_media, image=load_image_stop, command=pause_song)
button_stop["bg"] = "white"
button_stop["border"] = "0"

# Generation Options - Time Slider
slider_time = Scale(master=frame_options, bg="white", label="Song Length (seconds)", to=600, orient=tk.HORIZONTAL, length=300)
slider_time.grid(row=0, column=0)

time_value = IntVar()
time_entry = Entry(master=frame_options, width=10, textvariable=time_value)
time_entry.grid(row=0, column=1)

time_button = tk.Button(master=frame_options, bg="white", text="Set", height=1, width=3, command=set_time)
time_button.grid(row=0, column=2)

# Generation Options - Instrument Selection
radio_options = IntVar()

Radiobutton(master=frame_radio, text="Piano", variable=radio_options, value=1).grid(row=0, column=0, sticky=tk.W)
Radiobutton(master=frame_radio, text="Other", variable=radio_options, value=2).grid(row=0, column=1, sticky=tk.W)
Radiobutton(master=frame_radio, text="Other2", variable=radio_options, value=3).grid(row=0, column=2, sticky=tk.W)

# Window application loop
window.mainloop()