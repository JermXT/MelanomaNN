import nnOutput, impageprocOut, sys
total = len(sys.argv)
def run(total):
    if(total > 1):
        print "too many args"
        return
    const = 0.9
    answer = imageprocOut.run(total[0],const)
    while(answer == False):
        const = const+0.1
        answer = impageprocOut.run(total[0],const)
    print nnOutput.run(answer)
    
