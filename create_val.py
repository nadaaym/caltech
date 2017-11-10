import os
import shutil
import random

main_data='/home/nadaaym/caltech'

list_of_classes=os.listdir(main_data)
num=len(list_of_classes)
ref_copy=15
for i in range (0,num+1):
    newpath=r'/home/nadaaym/validation_data/%s' %(list_of_classes[i])
    os.makedirs(newpath)
    current=main_data + '/' + list_of_classes[i]
    for k in range(ref_copy):
        chosen_one = random.choice(os.listdir(current))
        print (chosen_one)
        file_to_move= current + '/' + chosen_one
        shutil.move(file_to_move, newpath)
        print(file_to_move)
print('Finished !')
