from tkinter import *
import pandas as pd

FILE = '/Users/niklastecklenburg/Desktop/QuestionAnswering/WOT_internal_refined.xlsx'
INDEX = 1032
DATA = pd.read_excel(FILE)

def Take_input(skip = False):
    global INDEX
    global DATA 
        
    if not skip:    
        DATA.at[INDEX, 'Annotated'] = context.get("1.0", END)
        
    DATA.at[INDEX, 'comment'] = comment.get("1.0", END)
        
    INDEX +=1
    if INDEX % 10 == 0:
        DATA.to_excel(FILE)
    print(INDEX)
        
    QUESTION.set(DATA.at[INDEX, 'question'])
    ANSWER.set(DATA.at[INDEX, 'answer_extr'])
    try:
        HIST0.set(DATA.at[INDEX, 'history'].split('|')[0])
    except:
        HIST0.set('')
    try:
        HIST1.set(DATA.at[INDEX, 'history'].split('|')[1])
    except:
        HIST1.set('')
    try:
        HIST2.set(DATA.at[INDEX, 'history'].split('|')[2])
    except:
        HIST2.set('')
    try:
        HIST3.set(DATA.at[INDEX, 'history'].split('|')[3])
    except:
        HIST3.set('')
    
    context.delete("1.0", END)
    comment.delete("1.0", END)
    context.insert(END, DATA.at[INDEX, 'Annotated'])
    comment.insert(END, DATA.at[INDEX, 'comment'])
    root.update()
        
    
root = Tk()
root.geometry("1200x600")
root.title("Labeling")

QUESTION = StringVar(root, value=DATA.at[INDEX, 'question'])
ANSWER = StringVar(root, value=DATA.at[INDEX, 'answer_extr'])

try:
    HIST0 = StringVar(root, value=DATA.at[INDEX, 'history'].split('|')[0])
except:
    HIST0 = StringVar(root, value='')
try:
    HIST1 = StringVar(root, value=DATA.at[INDEX, 'history'].split('|')[1])
except:
    HIST1 = StringVar(root, value='')
try:
    HIST2 = StringVar(root, value=DATA.at[INDEX, 'history'].split('|')[2])
except:
    HIST2 = StringVar(root, value='')
try:
    HIST3 = StringVar(root, value=DATA.at[INDEX, 'history'].split('|')[3])
except:
    HIST3 = StringVar(root, value='')

	
q = Label(text=f"Question: {QUESTION.get()}", textvariable=QUESTION)
a = Label(text=f"Answer: {ANSWER.get()}", textvariable=ANSWER)
h0 = Label(text=f"History: {HIST0.get()}", textvariable=HIST0)
h1 = Label(text=f"History: {HIST1.get()}", textvariable=HIST1)
h2 = Label(text=f"History: {HIST2.get()}", textvariable=HIST2)
h3 = Label(text=f"History: {HIST3.get()}", textvariable=HIST3)


context = Text(root,
               height = 20,
               width = 100,
               bg = "light yellow",)

context.insert(END, DATA.at[INDEX, 'Annotated'])

Ok = Button(root, height = 2,
            width = 20,
            text ="Ok",
            command = lambda:Take_input())

Remove = Button(root, 
                height = 2,
                width = 20,
                text ="Remove from df",
                command = lambda:Take_input(True))

comment = Text(root,
               height = 4,
               width = 50
               )
comment.insert(END, DATA.at[INDEX, 'comment'])

q.pack()
a.pack()
h0.pack()
h1.pack()
h2.pack()
h3.pack()
context.pack()
Ok.pack()
Remove.pack()
comment.pack()

mainloop()
DATA.to_excel(FILE)
