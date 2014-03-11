import pygraphviz
G = pygraphviz.AGraph(directed=True)
G.add_edge(0,'H')
e = G.get_edge(0,'H')
e.attr['label'] = '1/2'
e.attr['length'] = 5
G.edge_attr.update()
G.add_edge('H',0)
e = G.get_edge('H',0)
e.attr['label'] = '1/2'
G.add_edge(0,0)
e = G.get_edge(0,0)
e.attr['label'] = '1/2'
G.add_edge('H','HH')
e = G.get_edge('H','HH')
e.attr['label'] = '1/2'
G.layout(prog='dot')
G.draw('./Coin_MC.png')

import pygraphviz
G = pygraphviz.AGraph(directed=True)
G.add_edge('I','D')
e = G.get_edge('I','D')
e.attr['label'] = '&#955;(I,D)'
G.edge_attr.update()
G.add_edge('D','E')
e = G.get_edge('D','E')
e.attr['label'] = '&#955;(D,E)'
G.add_edge('D','D')
e = G.get_edge('D','D')
e.attr['label'] = '&#955;(D,D)'
G.layout(prog='dot')
G.draw('./PK_MC.png')

import pygraphviz
G = pygraphviz.AGraph(directed=True)
for i in ['A','T','C','G']:
    for j in ['A','T','C','G']:

        G.add_edge(i,j)
        e = G.get_edge(i,j)
        e.attr['label'] = '&#955;('+i+','+j+')'
        e.attr['weight'] = 10
        
G.layout(prog='circo')
G.draw('./ATCG_MC.png')

import pygraphviz
G = pygraphviz.AGraph(directed=True)
for i in ['A','B','C']:
    for j in ['A','B','C']:
        if (i =='A' and j == 'C') or\
            (i=='C' and j == 'A') or \
            (i==j):
                continue
        G.add_edge(i,j)
        e = G.get_edge(i,j)
        e.attr['label'] = '&#955;('+i+','+j+')'
        e.attr['weight'] = 10
        
G.layout(prog='dot')
G.draw('./ABC_MC.png')
