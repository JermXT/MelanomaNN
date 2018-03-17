import imageprocOut, sys
import nnOutput
#total = sys.argv
def run(total):
    #print total
    #if(len(total) > 2):
    #    print "too many args"
    #    return
    const = 1.0
    answer = imageprocOut.run(total,const)
    #while(answer == False):
    #    print "here"
    #    const = const-0.1
    #    answer = imageprocOut.run(total,const)
    answer = imageprocOut.run(total,const) 
    #print answer
    final = nnOutput.run([answer])
    final = final[0][0]
    print final
    if(final >=100):
        return True
    else:
        return False
#answer = run("Images/ISIC_0000016.png")
#print answer
