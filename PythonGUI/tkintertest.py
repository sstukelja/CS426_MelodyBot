import tkinter as tk
from tkinter import filedialog
from tkinter import Menu
import pygame

def get_file_name(x):
    start = x.rfind('/') + 1
    return x[start:len(x)]

def add_song():
    file_path = filedialog.askopenfilename()
    song_tuple = (get_file_name(file_path), file_path)
    list_songs.append(song_tuple)
    listbox_songs.insert(tk.END, song_tuple[0])
	
def play_song():
    pygame.mixer.init()
    pygame.mixer.music.load((list_songs[(listbox_songs.curselection()[0])])[1])
    pygame.mixer.music.play()

# Initilization of window application
window = tk.Tk()

window.title("MelodyBot")
window.geometry("1200x600")
window.configure(background="white")

# List of tuples containing file information
list_songs = list()

# Listbox containing user-facing song names
listbox_songs = tk.Listbox(master=window, width=50, selectmode=tk.BROWSE)
listbox_songs.pack()

# Menus
menu_bar = Menu(window)
window.configure(menu=menu_bar)

menu_file = Menu(menu_bar)
menu_edit = Menu(menu_bar)

menu_file.add_command(label="Import Song...",command=add_song)
menu_edit.add_command(label="Choose Settings...")

menu_bar.add_cascade(label="File", menu=menu_file)
menu_bar.add_cascade(label="Edit", menu=menu_edit)


# Buttons controlling music playback
load_image_play = tk.PhotoImage(file="C:/Users/Lepi/Desktop/CS425/Play.png")

button_play = tk.Button(master=window, image=load_image_play, command=play_song)
button_play["bg"] = "white"
button_play["border"] = "0"
button_play.pack()

# Window application loop
window.mainloop()