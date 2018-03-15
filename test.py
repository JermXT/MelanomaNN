import imageprocOut, sys
import nnOutput
#total = sys.argv
def run(total):
    #print total
    #if(len(total) > 2):
    #    print "too many args"
    #    return
    const = 0.9
    answer = imageprocOut.run(total,const)
    while(answer == False):
        print "here"
        const = const+0.1
        answer = impageprocOut.run(total,const)
    #print answer
    final = nnOutput.run([answer])
    final = final[0][0]
    if(final >=100):
        return True
    else:
        return False
#run("Images/ISIC_0000016.png")
