import imageprocOut, sys
import nnOutput
total = sys.argv
def run(total):
    print total
    if(len(total) > 2):
        print "too many args"
        return
    const = 0.9
    answer = imageprocOut.run(total[1],const)
    while(answer == False):
        const = const+0.1
        answer = impageprocOut.run(total[1],const)
    #print answer
    final[0][0] = nnOutput.run([answer])
    if(final >=100):
        return True
    else:
        return False
run(total)
