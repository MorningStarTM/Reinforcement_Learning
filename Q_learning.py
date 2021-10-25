import numpy as np
import matplotlib.pyplot as plt
from pylab import*
import networkx as nx
edges = [(0,1),(1,5),(5,6),(5,4),(1,2),
		 (1,3),(9,10),(2,4),(0,6),(6,7),
		 (8,9),(7,8),(1,7),(3,9)]
goal = 10
G = nx.Graph()
G.add_edges_from(edges)
pos = nx.spring_layout(G)
nx.draw_networkx_edges(G,pos)
nx.draw_networkx_nodes(G,pos)
nx.draw_networkx_labels(G,pos)


MATRIX_SIZE = 11
M = np.matrix(np.ones(shape=(MATRIX_SIZE,MATRIX_SIZE)))
M*=-1

for point in edges:
	print(point)
	if point[1] == goal:
		M[point] = 100
	else:
		M[point] = 0

	if point[0] == goal:
		M[point[::-1]] = 100
	else:
		M[point[::-1]] = 0

M[goal,goal] = 100
print(M)

gamma = 0.85
Q = np.matrix(np.zeros([MATRIX_SIZE,MATRIX_SIZE]))

initial_state = 0

#find possible path
def Possible_state(state):
	current_state = M[state, ]
	available_action = np.where(current_state >= 0)[1]
	return available_action

#Choose a path randomly
def next_actions(available_action_range):
	next_action = int(np.random.choice(available_action_range,1))
	return next_action

#Update the Q-matrix
def Update_Matrix(current_state, action, gamma):
	max_index = np.where(Q[action, ] == np.max(Q[action, ]))[1]
	if max_index.shape[0]>1:
		max_index = int(np.random.choice(max_index,1))
	else:
		max_index = int(max_index)
	max_value = Q[action,max_index]
	Q[current_state,action] = M[current_state,action] + gamma * max_value
	if np.max(Q) > 0:
		return np.sum(Q/np.max(Q)*100)
	else:
		return (Q)

'''available_action_range = Possible_state(initial_state)
next_action = next_actions(available_action_range)
u = Update_Matrix(next_action,1,0.5)
print(u)'''

#Training
scores = []
for i in range(1000):
	current_state = np.random.randint(0, int(Q.shape[0]))
	available_action_range = Possible_state(current_state)
	action = next_actions(available_action_range)
	score = Update_Matrix(current_state,action,0.85)
	scores.append(score)

print(Q/np.max(Q)*100)

#Testing
path = []
current_state = 6
while current_state!= 10:
	next_index = np.where(Q[current_state, ] == np.max(Q[current_state, ]))[1]
	if (next_index.shape[0]>1):
		next_index = int(np.random.choice(next_index,1))
	else:
		next_index = int(next_index)

	path.append(next_index)
	current_state = next_index

print(path)
plt.show()