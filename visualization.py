from google.colab import files
import pandas as pd
from networkx.algorithms.community import greedy_modularity_communities
#uploaded = files.upload()
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
test = plt.figure()
fig, ax = plt.subplots(figsize=(15, 15))

G = nx.Graph()
df =pd.read_csv('1.csv',error_bad_lines=False, engine="python")
#print(df['Node1'])
df1= list(df['Node1'])
#print(list(df1))
df2= list(df['Node2'])
relationships = pd.DataFrame({'from': df1, 
                              'to': df2})
overall = df1+df2
test_list = list(set(overall))
#print(test_list )
new =[]
for i in range(len(test_list)):
  new.append(test_list[i].join(x for x in test_list[i] if x.isalpha()))
#print(new)
# Create DF for node characteristics
carac = pd.DataFrame({'ID':test_list, 
                      'type':new})
# Create graph object
G = nx.from_pandas_edgelist(relationships, 'from', 'to', create_using=nx.Graph())

# Make types into categories
carac = carac.set_index('ID')
carac = carac.reindex(G.nodes())
carac['type'] = pd.Categorical(carac['type'])
carac['type'].cat.codes

node_shape=[]
# Set node shapes
for entry in carac.type:
  if entry =='F':
    node_shape.append(200)
  elif entry =='S':
    node_shape.append(200)
  else:
    node_shape.append(200)

communities = sorted(greedy_modularity_communities(G), key=len, reverse=True)

def set_node_community(G, communities):
    '''Add community to node attributes'''
    for c, v_c in enumerate(communities):
        for v in v_c:
            # Add 1 to save 0 for external edges
            G.nodes[v]['community'] = c + 1
            
def set_edge_community(G):
    '''Find internal edges and add their community to their attributes'''
    for v, w, in G.edges:
        if G.nodes[v]['community'] == G.nodes[w]['community']:
            # Internal edge, mark with community
            G.edges[v, w]['community'] = G.nodes[v]['community']
        else:
            # External edge, mark as 0
            G.edges[v, w]['community'] = 0

def get_color(i, r_off=1, g_off=1, b_off=1):
    r0, g0, b0 = 0, 0, 0
    n = 16
    low, high = 0.1, 0.9
    span = high - low
    r = low + span * (((i + r_off) * 3) % n) / (n - 1)
    g = low + span * (((i + g_off) * 5) % n) / (n - 1)
    b = low + span * (((i + b_off) * 7) % n) / (n - 1)
    return (r, g, b)

set_node_community(G, communities)
set_edge_community(G)

# Set community color for nodes
node_color = [get_color(G.nodes[v]['community']) for v in G.nodes]

# Set community color for internal edges
external = [(v, w) for v, w in G.edges if G.edges[v, w]['community'] == 0]
internal = [(v, w) for v, w in G.edges if G.edges[v, w]['community'] > 0]

internal_color = [get_color(G.edges[e]['community']) for e in internal]
                      
pos = nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=0.0001, center=None, dim=2, seed=None)
#pos = nx.spring_layout(G, k=5)
#nx.draw_networkx(
  #G, pos=pos, node_size=0, edge_color="#333333", alpha=0.5, with_labels=False)

# Draw graph
nx.draw_networkx(
    G, pos=pos, node_size=node_shape, edgelist=external, edge_color="#808080",node_color=node_color,
    alpha=0.6, with_labels=True)
# Draw internal edges
nx.draw_networkx(
    G, pos=pos, node_size=node_shape, edgelist=internal, node_color=node_color,edge_color=internal_color, alpha=1, with_labels=True)
plt.savefig('same_size_sep.pdf')
files.download('same_size_sep.pdf')
