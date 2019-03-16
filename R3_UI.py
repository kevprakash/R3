#-*- coding:utf-8 -*-

# import everything from tkinter
from tkinter import *
from tkinter import messagebox



#Create a close_window button
def close_window (my_window):
    my_window.destroy()


#Create a confirmation button.
def confirmation():
        MsgBox = messagebox.askquestion('Exit Application', 'Are you sure you want to exit the application',
                                           icon='warning')
        if MsgBox == 'yes':
            my_window.destroy()
        else:
            messagebox.showinfo('Return', 'You will now return to the application screen')


#Create Window object
my_window = Tk()
my_window.title("Recurrent Reward-Learning Reinforcement")




#Size of the window
width_of_window = 769
height_of_window = 588

#Depends on what DPI are you using
screen_width = my_window.winfo_screenwidth()
screen_heigth = my_window.winfo_screenheight()

#Calculate x and y coordinate
x_coordinate = (screen_width / 2) - (width_of_window / 2)
y_coordinate = (screen_heigth / 2) - (height_of_window / 2)

#Display the window in the screen
my_window.geometry("%dx%d+%d+%d" % (width_of_window, height_of_window, x_coordinate, y_coordinate))
my_window.resizable(width = False, height = False)           #Cannot resize the window


#Creat Frames/ Labels

frame_a = Frame(my_window,height = 155, width = 155)
frame_b = Frame(my_window,height = 152, width = 152)
frame_c = Frame(my_window,height = 152, width = 152)
frame_d = Frame(my_window,height = 152, width = 152)

frame_a.grid(row = 0, column = 0)
frame_b.grid(row = 0, column = 1)
frame_c.grid(row = 1, column = 0)
frame_d.grid(row = 1, column = 1)


label1 = Label(frame_a,
               text = "Introduction:\nLine 1\nLine 2\nLine 3\nLine 4\nLine 5\nLine 6",
               #bg = '#faff9b',
               fg = '#5b112c',
               font = "Verdana 20 bold italic",
               relief = 'flat',
               width = 27,
               height = 12,
               anchor = NW)
label1.pack()

label2 = Label(frame_c,
               text = "USER INPUT BOX \n Please input your text below box.",
               #bg='#faff9b',
               fg='#5b112c',
               font="Verdana 20 bold italic",
               relief='flat',
               width=27,
               height=12,
               anchor= N

               )
#Create ScrollBar

scrollbar = Scrollbar(frame_c)
scrollbar.pack(side = RIGHT, fill = Y)
label2_entry = Text(my_window, width = "45", height = "10", yscrollcommand = scrollbar.set)


for line in range(100):
    label2_entry.insert(END,'' + str())
    label2_entry.pack(side = LEFT, fill = BOTH)
    scrollbar.config(command = label2_entry.yview)

label2_entry.grid(row = 1, column = 0)
label2.grid(row = 0, column = 0)
label2.pack(side = TOP)


label3 = Label(frame_d,
               text = "FAQ:",
               #bg='#faff9b',
               fg='red',
               font="Verdana 20 bold italic",
               relief='flat',
               width=27,
               height=12,
               anchor=NW
               )
label3.grid(row = 0, column = 0)
label3.pack()


#Create Importing Buttons
button_1 = Button(frame_b,
                  text = "TRAIN",
                  width = "18",
                  )
button_1.grid(row = 0, column = 0)
button_1.pack()

button_2 = Button(frame_b,
                  text = "SAVE",
                  width = "18",
                  )
button_2.grid(row = 1, column = 0)
button_2.pack()


button_3 = Button(frame_b,
                  text = "LOAD",
                  width = "18",
                  )
button_3.grid(row = 1, column = 0)
button_3.pack()

button_4 = Button(frame_b,
                  text = "PAUSE",
                  width = "18",
                  )
button_4.grid(row = 1, column = 0)
button_4.pack()

button_5 = Button(frame_b,
                  text = "QUIT",
                  width = "18",
                  comman = my_window.destroy
                  )
button_5.grid(row = 1, column = 0)
button_5.pack()
button_5.config(command = confirmation)

my_window.mainloop()
