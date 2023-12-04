import numpy as np
import matplotlib.pyplot as plt
import random

def initialize_lattice(size):
    """ Initialize an empty lattice with all sites unoccupied (cluster 0). """
    return np.zeros((size, size), dtype=int)

def occupy_site(lattice, occupation_prob,size):
    """ Occupy a site in the lattice based on the occupation probability. """
    size = len(lattice)
    for trials in range(size**2):  # Limit the number of attempts
        x, y = random.randint(0, size - 1), random.randint(0, size - 1)
        if lattice[x, y] == 0 and random.random() < occupation_prob:
            return x, y
    return False,False

def pos_chk(dx,dy,size):
    if 0 <= dx < size and  0 <= dy < size:
        return True

def neighbour_test(i,j,size,lattice):
    neigh = [(i-1,j),(i+1,j),(i,j-1),(i,j+1)]
    cluster = []
    for dx,dy in neigh:
        if pos_chk(dx,dy,size):
            if lattice[dx,dy] != 0:
                cluster.append(lattice[dx,dy])
    return cluster

def check_for_spanning_cluster(clusters_dic, size):
    for cluster_id, coordinates in clusters_dic.items():
        has_top = has_bottom = has_left = has_right = False

        for x, y in coordinates:
            if x == 0:
                has_top = True
            elif x == size - 1:
                has_bottom = True
            if y == 0:
                has_left = True
            elif y == size - 1:
                has_right = True

        # Check if the cluster spans top to bottom and left to right
        if has_top and has_bottom and has_left and has_right:
            return True

    return False

            

def percolation(lattice, occupation_prob, size):
    """ Form a spanning cluster in the lattice. """
    current_cluster = 1
    clusters_dic = {}

    while True:
        x, y = occupy_site(lattice, occupation_prob,size)
        # if x is False:
        #     print("NOOOOOOOO")
        #     return lattice, clusters_dic
            
        cluster = neighbour_test(x,y,size,lattice)
        if len(cluster) == 0:
            lattice[x,y] = current_cluster
            if current_cluster not in clusters_dic:
                clusters_dic[current_cluster] = [(x,y)]
            else:
                clusters_dic[current_cluster].append((x,y))
            current_cluster = current_cluster +1
            print(clusters_dic)
        if len(cluster) == 1:
            lattice[x,y] = cluster[0]
            clusters_dic[cluster[0]].append((x,y))
            print(clusters_dic)
        
        if len(cluster) > 1:
            min_value = min(cluster)
            lattice[x,y] = min_value
            cluster.remove(min_value)
            
            transition_list = []  # List to store transferred values

            for item in cluster:
                if item in clusters_dic:
                    transition_list.extend(clusters_dic[item])
                    print(clusters_dic)
                    del clusters_dic[item]
                    
            for dx,dy in transition_list:
                lattice[dx,dy] = min_value
                clusters_dic[min_value].append((dx,dy))
                
        if check_for_spanning_cluster(clusters_dic, size):
            print("Spanning cluster found!")
            break
                
    return lattice, clusters_dic
        

                
if __name__ == "__main__":
    # Initialize parameters
    size = 5
    occupation_prob = 0.2  # Probability of a site being occupied

    # Initialize lattice and form a spanning cluster
    lattice = initialize_lattice(size)
    lattice,clusters_dic = percolation(lattice, occupation_prob, size)
    print(clusters_dic)
    
    plt.imshow(lattice)
    plt.show()

   


# if len(cluster) > 1:
        #     min_value = min(cluster)
        #     lattice[x, y] = min_value

        #     # Ensure clusters_dic has an entry for min_value
        #     if min_value not in clusters_dic:
        #         clusters_dic[min_value] = [(x, y)]
        #     else:
        #         clusters_dic[min_value].append((x, y))

        #     # Merge clusters
        #     for item in cluster:
        #         if item != min_value and item in clusters_dic:
        #             for coord in clusters_dic[item]:
        #                 lattice[coord[0], coord[1]] = min_value
        #                 if coord not in clusters_dic[min_value]:
        #                     clusters_dic[min_value].append(coord)
        #             del clusters_dic[item]
                
        
        # if len(cluster) > 1:
        #     min_value = min(cluster)
        #     lattice[x, y] = min_value

        #     # Ensure clusters_dic has an entry for min_value
        #     if min_value not in clusters_dic:
        #         clusters_dic[min_value] = [(x, y)]
        #     else:
        #         clusters_dic[min_value].append((x, y))

        #     # Merge clusters
        #     for item in cluster:
        #         if item != min_value and item in clusters_dic:
        #             for coord in clusters_dic[item]:
        #                 lattice[coord[0], coord[1]] = min_value
        #                 if coord not in clusters_dic[min_value]:
        #                     clusters_dic[min_value].append(coord)
        #             del clusters_dic[item]