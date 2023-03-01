from tkinter import *
import pandas as pd

FILE = 'revisit2.csv'
INDEX = 0
DATA = pd.read_csv(FILE)

def Take_input(skip = False):
    global INDEX
    global DATA 
        
    if not skip:    
        DATA.at[INDEX, 'Annotated'] = context.get("1.0", END)
        
    DATA.at[INDEX, 'comment'] = comment.get("1.0", END)
        
    INDEX +=1
    if INDEX % 10 == 0:
        DATA.to_csv(FILE)
        print(INDEX)
        
    QUESTION.set(DATA.at[INDEX, 'question'])
    ANSWER.set(DATA.at[INDEX, 'answer'])
    context.delete("1.0", END)
    comment.delete("1.0", END)
    context.insert(END, DATA.at[INDEX, 'Context'])
    root.update()
        
    
root = Tk()
root.geometry("1200x600")
root.title("Labeling")

QUESTION = StringVar(root, value=DATA.at[INDEX, 'question'])
ANSWER = StringVar(root, value=DATA.at[INDEX, 'answer'])
	
q = Label(text=f"Question: {QUESTION.get()}", textvariable=QUESTION)
a = Label(text=f"Answer: {ANSWER.get()}", textvariable=ANSWER)

context = Text(root,
               height = 20,
               width = 100,
               bg = "light yellow",)

context.insert(END, DATA.at[INDEX, 'Context'])

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

q.pack()
a.pack()
context.pack()
Ok.pack()
Remove.pack()
comment.pack()

mainloop()
DATA.to_csv(FILE)
