from __future__ import division
#pydealer can be found at : http://pydealer.readthedocs.org/en/latest/index.html
import pydealer


#numin is the number of times "inside" was the correct option, numout is of course the reverse
numin = 0
numout = 0
deck = pydealer.Deck(re_shuffle = True)
deck.shuffle(times= 100) #totally arbitrary number, but one "shuffle" does not randomize the deck
current = deck.deal(2)   
next = deck.deal(1,rebuild=True)

for i in range (0,1000): # the number of times we repeat is arbitrary and I encourage you to change it  

   # deck.shuffle(times= 52)
    
    greater = current[0] if (current[0]).ge(current[1]) else current[1] #ternary statements, if the cards are the same these should still hold
    lesser = current[0] if (current[0]).le(current[1]) else current[1]

    
    #print "Greater - >  " , greater , "  Lesser ->  " , lesser , "  Next ->  " , next    
    if(next[0].gt(greater) or next[0].lt(lesser)):
        numout+=1
    elif(next[0].lt(greater) and next[0].gt(lesser)):
        numin+=1
    else:
        numout+=1
        numin+=1
        
    current[1] = current[0]
    current[0] = next[0]
    next = deck.deal(1,rebuild=True)

print "out ->" , numout
print "in ->" , numin
print "ratio out/in -> " , numout/numin


