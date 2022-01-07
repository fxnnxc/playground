
def dijkstra(graph, start, start_length):
    max_distance = 0
    distances = {key:0 for key in graph.keys()} 
    distances["{start}_L"] = start_length
    distances["{start}_R"] = start_length
    for key1 in graph.keys():
        pass 
        for key2 in graph.keys():
            pass 
            if distance[key1] 

    
    return max_distance 
    

def makegraph(nodes):
    graph = {}
    for i in range(len(nodes)):
        graph[f"{i}_L"] = {"v":nodes[i][0], "links":[]}
        graph[f"{i}_R"] = {"v":nodes[i][1], "links":[]}
        
    # from i to j
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i ==j :
                continue
            # left for i
            if nodes[i][0] == nodes[j][1]:
                graph[f"{i}_L"]['links'].append([f"{j}_R",  nodes[j][2]])
            if nodes[i][1] == nodes[j][0]:
                graph[f"{i}_R"]['links'].append([f"{j}_R",  nodes[i][2]])
        
    return graph 
    
def main():
    nodes = []
    a = input() 
    for i in range(int(a)):
        values, distance = input().split()
        v0,v1 = list(values)
        nodes.append([v0,v1, int(distance)])
    graph = makegraph(nodes)
    print(graph)
    distance = dijkstra(graph)
    print(distance)        
    
    
main()

        