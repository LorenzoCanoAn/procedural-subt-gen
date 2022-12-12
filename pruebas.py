import matplotlib.pyplot as plt

print(plt.gca())

fig = plt.figure()
axis = fig.add_subfigure()

print(plt.gca())