import numpy as np
from gensim.models import KeyedVectors
nor = np.linalg.norm
import warnings
warnings.filterwarnings('ignore')
import tkinter as tk
import tkinter.ttk as ttk

# load vector data : both cbow and skip-gram
model1 = KeyedVectors.load_word2vec_format('./cbow30.bin', unicode_errors='ignore', binary=True)
model2 = KeyedVectors.load_word2vec_format('./skip20.bin', unicode_errors='ignore', binary=True)

# callback for Enter key
def enter(event):
    a = radio_value.get()
    if a == 0:
        word = form1.get()
        if word == '':
            r1.set('enter a word in the form')
            r2.set('')
        elif word not in model1.wv.vocab:
            r1.set('{} not in the vocabulary'.format(word))
            r2.set('')
        else:
            result1 = model1.wv.most_similar(positive=[word], topn=15)
            result2 = model2.wv.most_similar(positive=[word], topn=15)
            str1, str2 = 'model1:CBOW', 'model2:skip-gram'
            i, j = 0, 0
            for result in result1:
                if result[0][0] not in '.!?:;"' and i < 5:
                    str1 += '\n{}\t{}'.format(result[0], round(result[1], 4))
                    i += 1
            for result in result2:
                if result[0][0] not in '.!?:;"' and j < 5:
                    str2 += '\n{}\t{}'.format(result[0], round(result[1], 4))
                    j += 1
            r1.set(str1)
            r2.set(str2)

    elif a == 1:
        word1 = form1.get()
        word2 = form2.get()
        if word1 == '' or word2 == '':
            r1.set('enter 2 words in the form')
            r2.set('')
        elif word1 not in model1.wv.vocab:
            r1.set('{} not in the vocabulary'.format(word1))
            r2.set('')
        elif word2 not in model1.wv.vocab:
            r1.set('{} not in the vocabulary'.format(word2))
            r2.set('')
        else:
            vec1_c = model1.wv[word1]  # cbow
            vec2_c = model1.wv[word2]
            vec1_s = model2.wv[word1]  # skip-gram
            vec2_s = model2.wv[word2]
            result1 = round(float(np.dot(vec1_c, vec2_c)) / (nor(vec1_c) * nor(vec2_c)), 4)
            result2 = round(float(np.dot(vec1_s, vec2_s)) / (nor(vec1_s) * nor(vec2_s)), 4)
            r1.set('model1: ' + str(result1))
            r2.set('model2: ' + str(result2))

    elif a == 2:
        word1 = form1.get()
        word2 = form2.get()
        word3 = form3.get()
        if word1 == '' or word2 == '' or word3 == '':
            r1.set('enter 3 words in the form')
            r2.set('')
        elif word1 not in model1.wv.vocab:
            r1.set('{} not in the vocabulary'.format(word1))
            r2.set('')
        elif word2 not in model1.wv.vocab:
            r1.set('{} not in the vocabulary'.format(word2))
            r2.set('')
        elif word3 not in model1.wv.vocab:
            r1.set('{} not in the vocabulary'.format(word3))
            r2.set('')
        else:
            result1 = model1.wv.most_similar(positive=[word1, word3], negative=[word2], topn=5)
            result2 = model2.wv.most_similar(positive=[word1, word3], negative=[word2], topn=5)
            str1, str2 = 'model1:CBOW', 'model2:skip-gram'
            for result in result1:
                str1 += '\n{}\t{}'.format(result[0], round(result[1], 4))
            for result in result2:
                str2 += '\n{}\t{}'.format(result[0], round(result[1], 4))
            r1.set(str1)
            r2.set(str2)


# mode change
def button1(event):
    label_text.set('enter a word')
    f1.set('')
    form2.grid_forget()
    form3.grid_forget()
    r1.set('')
    r2.set('')

def button2(event):
    label_text.set('enter 2 words')
    f1.set('')
    f2.set('')
    label1.grid_forget()
    label2.grid_forget()
    form3.grid_forget()
    form2.grid(row=5, column=0, columnspan=2)
    label1.grid(row=6, column=0, padx=3, pady=3, sticky=tk.W + tk.E)
    label2.grid(row=6, column=1, padx=3, pady=3, sticky=tk.W + tk.E)
    r1.set('')
    r2.set('')

def button3(event):
    label_text.set('enter 3 words')
    f1.set('')
    f2.set('')
    f3.set('')
    label1.grid_forget()
    label2.grid_forget()
    form2.grid(row=5, column=0, columnspan=2)
    form3.grid(row=6, column=0, columnspan=2)
    label1.grid(row=7, column=0, padx=3, pady=3, sticky=tk.W + tk.E)
    label2.grid(row=7, column=1, padx=3, pady=3, sticky=tk.W + tk.E)
    r1.set('')
    r2.set('')


root = tk.Tk()
#root.geometry('750x500+300+100')
root.title("Similarity Searcher")

# radio button
radio_value = tk.IntVar()
radio_value.set(0)
mode1 = 'search mode: gives the most similar 5 words'
mode2 = 'compare mode: gives the cosine similarity of two words'
mode3 = 'calculate mode: gives the result of word1 - word2 + word3'
radio1 = tk.Radiobutton(root, text=mode1, variable=radio_value, value=0)
radio1.grid(row=0, column=0, columnspan=2)
radio2 = tk.Radiobutton(root, text=mode2, variable=radio_value, value=1)
radio2.grid(row=1, column=0, columnspan=2)
radio3 = tk.Radiobutton(root, text=mode3, variable=radio_value, value=2)
radio3.grid(row=2, column=0, columnspan=2)

# label
label_text = tk.StringVar()
label_text.set('enter a word')
ttk.Label(root, textvariable=label_text, font=('THSarabun','20')).grid(row=3, column=0, columnspan=2)

# result label
r1 = tk.StringVar()
r1.set('')
r2 = tk.StringVar()
r2.set('')

# form
f1 = tk.StringVar()
f1.set('')
f2 = tk.StringVar()
f2.set('')
f3 = tk.StringVar()
f3.set('')


form1 = ttk.Entry(root, justify='center', textvariable=f1, font=('THSarabun','40'))
form1.grid(row=4, column=0, columnspan=2)
form2 = ttk.Entry(root, justify='center', textvariable=f2, font=('THSarabun','40'))
form2.grid(row=5, column=0, columnspan=2)
form2.grid_forget()
form3 = ttk.Entry(root, justify='center', textvariable=f3, font=('THSarabun','40'))
form3.grid(row=6, column=0, columnspan=2)
form3.grid_forget()
label1 = ttk.Label(root, textvariable=r1, font=('THSarabun','20'))
label1.grid(row=7, column=0, padx=3, pady=3, sticky=tk.W+tk.E)
label2 = ttk.Label(root, textvariable=r2, font=('THSarabun','20'))
label2.grid(row=7, column=1, padx=3, pady=3, sticky=tk.W+tk.E)



# press enter key
root.bind_all('<Return>', enter)

# press radio button
radio1.bind('<Button-1>', button1)
radio2.bind('<Button-1>', button2)
radio3.bind('<Button-1>', button3)

root.mainloop()