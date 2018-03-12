import os, imageprocOut
import pickle
list = os.listdir("safe")
input = []
output = []
#print list
if('.DS_Store' in list):
    list.remove('.DS_Store')
#f=open('benign.txt','w')
for x in range(len(list)):
    output.append(0.1)
    answer = imageprocOut.run("safe/"+list[x],1.1)
    input.append(answer)
    print x+1
    print input
with open('benign.txt', 'wb') as fp:
    pickle.dump(input, fp)
print output
print input
#f.write(output)
#f.close()
