import numpy as np

def transform_index_to_2D(indexes,points=10):
    column = []
    row = []
    for k in indexes:
        row.append(k//(points+1))
        column.append((k%(points+1)))
    return row,column

def indices_threshhold(p=0.97,probabilities=None,points=10):
    if probabilities is None:
        probabilities = np.loadtxt('probability.txt')
        probabilities = probabilities / np.sum(probabilities)
    flatten = probabilities.reshape(-1)
    indices = np.argsort(-flatten)
    """sorted is from smallest prob to largest"""
    sorted = np.sort(flatten)
    cur_value = 1
    reached_threshold = False
    cur_index = 0
    while not reached_threshold:
        cur_value -= sorted[cur_index]
        cur_index += 1
        if cur_value <= p:
            reached_threshold = True
    rows,columns=transform_index_to_2D(indexes=indices[:(points+1)**2-cur_index])
    return rows,columns,indices[(points+1)**2-cur_index:],probabilities