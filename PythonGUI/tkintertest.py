import tkinter as tk
from tkinter import filedialog
from tkinter import Menu
from tkinter import Scale
from tkinter import Frame
from tkinter import IntVar
from tkinter import Entry
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
window.configure(background="white")

pygame.mixer.init()

# List of tuples containing file information
list_songs = list()

# Listbox containing user-facing song names
listbox_songs = tk.Listbox(master=window, width=50, selectmode=tk.BROWSE)
listbox_songs.grid(row=0, column=0)

# Menus
menu_bar = Menu(window)
window.configure(menu=menu_bar)

menu_file = Menu(menu_bar)
menu_edit = Menu(menu_bar)

menu_file.add_command(label="Import Song...",command=add_song)
menu_edit.add_command(label="Choose Settings...")

menu_bar.add_cascade(label="File", menu=menu_file)
menu_bar.add_cascade(label="Edit", menu=menu_edit)

frame_options = Frame(master=window, width=200, height=100)
frame_options.grid(row=0, column=1)
# Buttons controlling music playback

#Play button
load_image_play = tk.PhotoImage(file="C:/Users/Lepi/Desktop/CS425/PythonGUI/Play.png")
button_play = tk.Button(master=window, image=load_image_play, command=play_song)
button_play["bg"] = "white"
button_play["border"] = "0"
button_play.grid(row=1, column=0)

#Stop button
load_image_stop = tk.PhotoImage(file="C:/Users/Lepi/Desktop/CS425/PythonGUI/Pause.png")
button_stop = tk.Button(master=window, image=load_image_stop, command=pause_song)
button_stop["bg"] = "white"
button_stop["border"] = "0"

# Generation Options
slider_time = Scale(master=frame_options, bg="white", label="Song Length (seconds)", to=600, orient=tk.HORIZONTAL, length=300)
slider_time.grid(row=0, column=0)

time_value = IntVar()
time_entry = Entry(master=frame_options, width=10, textvariable=time_value)
time_entry.grid(row=0, column=1)

time_button = tk.Button(master=frame_options, bg="white", text="Set", height=1, width=3, command=set_time)
time_button.grid(row=0, column=2)

# Window application loop
window.mainloop()