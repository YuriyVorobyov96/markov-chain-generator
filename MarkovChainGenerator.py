#!/usr/bin/env python
# coding: utf-8

# In[59]:


def markov_chain(
    states=['XSS','SQL','PHP'],
    transition_name=[['XX','XS','XP'],['SX','SS','SP'],['PX','PS','PP']],
    transition_matrix=[[0.6,0.3,0.1],[0.5,0.4,0.1],[0.6,0.2,0.2]],
    start_state='XSS',
    end_state='SQL',
    iterations_count=10000,
    steps_count=2,
    lang='eng',
    is_table=False,
    is_chain=False
):
    """
    Creating a Markov chain and calculating the probability of states.
    
    Keyword arguments:
    states -- Array of possible states
    transition_name -- Array of transitions
    transition_matrix -- Array of transitions probability
    start_state -- Start state
    end_state -- End state
    iterations_count -- Number of iterations
    steps_count -- Number of steps
    lang -- Language
    is_table -- Conclusion of the table of transitions with probabilistic values
    """
    import numpy as np
    import random as rm
    import pandas as pd
    import pydot
    import networkx as nx
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from functools import reduce
    
    if len(states) != len(transition_name) or len(states) != len(transition_matrix) or len(transition_matrix) != len(transition_name):
        raise Exception('Check the lists of states and transitions. Their lengths should be equal.')
    
    possibility_ckeck = 0
    cheker = 0
    while cheker < len(transition_matrix):
        possibility_ckeck += sum(transition_matrix[cheker])
        cheker += 1
    if possibility_ckeck != len(transition_matrix):
        raise Exception('Transition Matrix compiled incorrectly.')

    G = nx.DiGraph()

    def states_forecast(steps_count):
        state_type = start_state
        states_list = [state_type]
        step = 0
        prob = 1

        while step != steps_count:
            for i in range(len(states)):
                if is_chain == True:
                    G.add_node(states[i], pos=(rm.random() * 100, rm.random() * 100))
                if state_type == states[i]:
                    change = np.random.choice(transition_name[i],replace=True,p=transition_matrix[i])
                    for j in range(len(transition_name)):
                        if is_chain == True:
                            G.add_edge(states[i], states[j], weight=transition_matrix[i][j], arrowstyle='->', arrowsize=15, width=3)
                        if change == transition_name[i][j]:
                            prob = prob * transition_matrix[i][j]
                            state_type = states[j]
                            states_list.append(states[j])

            step += 1    
        return states_list

    list_states = []
    count = 0

    for iterations in range(1,iterations_count):
            list_states.append(states_forecast(steps_count))

    for smaller_list in list_states:
        if(smaller_list[2] == end_state):
            count += 1

    # считаем процентики
    percentage = (count/iterations_count) * 100
    

    if lang == 'rus':
        print("Вероятность начала в состоянии:{0} и конца в состоянии:{1}= ".format(start_state, end_state) + str(percentage) + '%')
    else:
        print("The probability of starting at state:{0} and ending at state:{1}= ".format(start_state, end_state) + str(percentage) + '%')
    if is_table == True:
        df = pd.DataFrame(
            transition_matrix,
            columns=states,
            index=states,
        )
        print('----')
        print(df)
        print('----')

    if is_chain == True:    
        pos = nx.get_node_attributes(G, 'pos')
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx(G, pos, node_color = 'w')
        nx.draw_networkx_edge_labels(G, pos, edge_labels = labels, node_size = 500)
        ax = plt.gca()
        ax.set_axis_off()
        plt.show()


# In[62]:


markov_chain(
['XSS','SQL','PHP'],
[['XX','XS','XP'],['SX','SS','SP'],['PX','PS','PP']],
[[0.6,0.3,0.1],[0.5,0.4,0.1],[0.6,0.2,0.2]],
'XSS',
'SQL',
is_table=True,
is_chain=True
)


# In[ ]:





# In[ ]:




