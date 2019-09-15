# Plots a specific hysteris function, like that of a schmitt trigger

x = [ 0, .55 , .55]
y = [1, 1 ,0]

x2 = [1, .45, .45]
y2 = [0, 0, 1]

l = plt.plot(x,y,color='k')[0]
add_arrow(l)
#add_arrow(l,position=0)


l2 = plt.plot(x2,y2,color='k')[0]
add_arrow(l2)

plt.title('Hysteris Function')
plt.xlabel('Input')
plt.ylabel('Output')
#add_arrow(l2,pos
