import matplotlib.pyplot as plt
import numpy as np

x=[] #np.linspace(0, 0, 0)
y=[] #x*2

# You probably won't need this if you're embedding things in a tkinter plot...
plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot(x, y, 'b-') # Returns a tuple of line objects, thus the comma

for phase in np.linspace(0, 10*np.pi, 500):
    x = np.append(x,len(x)+1)
    y=x*2
    # update ax.viewLim using the new dataLim

    #line1.set_ydata(np.sin(x + phase))
    line1.set_xdata(x)
    line1.set_ydata(y)

    ax.autoscale_view()
    ax.relim()

    print(phase)
    fig.canvas.draw()
    fig.canvas.flush_events()