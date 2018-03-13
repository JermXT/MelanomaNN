import os, imageprocOut
import pickle
list = os.listdir("ISIC_melanoma")
input = []
output = []
#print list
if('.DS_Store' in list):
    list.remove('.DS_Store')
#f=open('benign.txt','w')
for x in range(len(list)):
    output.append(0.9)
    answer = imageprocOut.run("ISIC_melanoma/"+list[x],1.2)
    input.append(answer)
    print x+1
    print input
with open('mela.txt', 'wb') as fp:
    pickle.dump(input, fp)
print output
print input
#f.write(output)
#f.close()
