from numba.typed import List
import numba as nb
import numpy as np
import open3d as o3d
from scipy.linalg import polar

def create_empty_nested_list(size):
    res=List()
    for itm in range(size):
        nested=List([0])
        nested.pop()
        res.append(nested)
    return(res)

class LatticeBuilderNode:
    """
    Builds OctTree for particle simulation
    """

    def __init__(self, is_leaf=False, parent=None, mass=1):
        self.is_leaf = is_leaf
        self.children = np.full((2, 2, 2), None, dtype=np.object)
        self.parent = parent
        self.mass = mass
        self.region_count = 0

    def make_children(self, filled):
        if not self.is_leaf:
            raise Exception("Cannot fill children because is not leaf node")
        self.is_leaf = False

        if(filled):
            for x in range(2):
                for y in range(2):
                    for z in range(2):
                        self.children[(x, y, z)] = LatticeBuilderNode(
                            is_leaf=True, parent=self)

        else:
            self.children = np.full((2, 2, 2), None, dtype=np.object)

    def set(self, point, depth, filled):  # -> is_empty (only if filled = false)
        """
        special behavior:
        if setting filled and target is too coarse, new nodes are created, even if it is already filled:

        |   |* *|    |   |*|*|
        |-------| -> |-------|
        |   |   |    |   |   |

        This does not apply to filling empty because particles are only made on filled ones
        """
        if depth < 0:
            raise Exception("Invalid depth of " + len(depth))
        if depth == 0:
            raise Exception(
                "Cannot set current node (can only set from parent)")
        target_node_key = tuple(np.array(point * 2, dtype=np.int_))
        target_node_pos = (point * 2) % 1
        if filled:
            if self.is_leaf:
                self.make_children(True)
            if depth == 1:
                self.children[target_node_key] = LatticeBuilderNode(
                    is_leaf=True, parent=self)
            else:
                if(self.children[target_node_key] is None):
                    self.children[target_node_key] = LatticeBuilderNode(
                        is_leaf=True, parent=self)
                    self.children[target_node_key].make_children(False)
                self.children[target_node_key].set(
                    target_node_pos, depth - 1, filled)
        else:
            if self.is_leaf:
                raise Exception("cannot delete leaf (need parent)")
            if depth == 1:
                self.children[target_node_key] = None
            else:
                if(self.children[target_node_key] is not None):
                    if self.children[target_node_key].is_leaf:
                        self.children[target_node_key].make_children(True)

                    if(self.children[target_node_key].set(target_node_pos, depth - 1, filled)):
                        self.children[target_node_key] = None
            return(np.all(self.children == None))
    def set_positions(self,lower,upper):
        if(self.is_leaf):
            self.position=(lower+upper)/2
            return
        self.position_lower=lower
        self.position_upper=upper

        template=[lower,(lower+upper)/2,upper]
        for x in range(2):
            for y in range(2):
                for z in range(2):
                    if(self.children[(x,y,z)] is None):
                        continue
                    self.children[(x,y,z)].set_positions(
                        np.array((template[x][0],template[y][1],template[z][2])),
                        np.array((template[x+1][0],template[y+1][1],template[z+1][2]))
                    )
    #   lower       upper
    #   |----------|       (2)
    #      [-----]
    #      a    b
    def overlap(lower, upper, a, b):# 0 no overlap, 1 some overlap, 2 all overlap
        if((lower <= a) and (b<=upper)):
            return 2
        elif (a<=upper<=b) or (a<=lower<=b):
            return 1
        else:
            return 0

    def get_AABB_nodes(self,lower,upper):
        template=[self.position_lower,(self.position_lower+self.position_upper)/2,self.position_upper]

        for x in range(2):
            xo=LatticeBuilderNode.overlap(lower[0],upper[0],template[x][0],template[x+1][0])
            for y in range(2):
                yo=LatticeBuilderNode.overlap(lower[1],upper[1],template[y][1],template[y+1][1])
                for z in range(2):
                    zo=LatticeBuilderNode.overlap(lower[2],upper[2],template[z][2],template[z+1][2])
                    o=min(xo,yo,zo)
                    if(o==0):
                        pass
                    elif(o==1):
                        if self.children[(x,y,z)] is not None:
                            if(self.children[(x,y,z)].is_leaf):
                                yield self.children[(x,y,z)]
                            else:
                                yield from self.children[(x,y,z)].get_AABB_nodes(lower,upper)
                    elif(o==2):
                        if self.children[(x,y,z)] is not None:
                            yield self.children[(x,y,z)]
    def get_AABB_leaf_nodes(self,lower,upper):
        for node in self.get_AABB_nodes(lower,upper):
            yield from node.get_leaf_nodes()
    def get_AABB_subtree(self,lower,upper):
        return SubTree(nodes=list(self.get_AABB_nodes(lower,upper)))

    def get_leaf_nodes(self):
        if(self.is_leaf):
            yield self
            return

        for x in range(2):
            for y in range(2):
                for z in range(2):
                    if(self.children[(x,y,z)] is not None):
                        yield from self.children[(x,y,z)].get_leaf_nodes()


    def addToArray(self,array):
        self.array_indx=len(array)
        array.append(self)
        if(not self.is_leaf):
            for x in range(2):
                for y in range(2):
                    for z in range(2):
                        if(self.children[(x,y,z)] is not None):
                            self.children[(x,y,z)].addToArray(array)
    def addSumOrder(self,summands,targets):
        if(self.is_leaf):
            return
        childrenIds=[]
        if(not self.is_leaf):
            for x in range(2):
                for y in range(2):
                    for z in range(2):
                        if(self.children[(x,y,z)] is None):
                            childrenIds.append(-1)
                        else:
                            self.children[(x,y,z)].addSumOrder(summands,targets)
                            childrenIds.append(self.children[(x,y,z)].array_indx)
        targets.append(self.array_indx)
        summands.append(childrenIds)

    #                                               region_filter(center, candidates(generator) )-> region (generator)
    def buildParticleArrays(self,lower,upper,region_aabb,region_filter):
        self.set_positions(lower,upper)

        #get tree nodes
        nodes=[]
        self.addToArray(nodes)
        node_count=len(nodes)

        #get efficient sum order from tree
        _sum_order_summands=[]
        _sum_order_targets=[]
        self.addSumOrder(_sum_order_summands,_sum_order_targets)
        sum_order_summands=np.array(_sum_order_summands)
        sum_order_targets=np.array(_sum_order_targets)
        #get leaf nodes of tree
        leaf_nodes=list(self.get_leaf_nodes())
        leaf_node_ids=np.array([leaf_node.array_indx for leaf_node in leaf_nodes])

        #copy initial particle position information to array for efficient processing
        particle_initial_positions=np.empty((node_count,3),dtype=np.float)
        for leaf_node in leaf_nodes:
            particle_initial_positions[leaf_node.array_indx]=leaf_node.position

        # initial particle position = particle positions
        particle_positions=np.array(particle_initial_positions)

        #copy particle mass information to array for efficient processing
        particle_masses=np.empty((node_count),dtype=np.float)
        for leaf_node in leaf_nodes:
            particle_masses[leaf_node.array_indx]=leaf_node.mass

        #calculate list of particles for each region
        region_particles=np.empty((node_count),dtype=np.object)

        for leaf_node in leaf_nodes:
            region_particles[leaf_node.array_indx]=[
                node.array_indx
                for node in region_filter(leaf_node,self.get_AABB_leaf_nodes(leaf_node.position-region_aabb,leaf_node.position+region_aabb))
            ]


        #create list for regions each particle is in
        particle_regions=np.empty((node_count),dtype=np.object)
        for leaf_node_id in leaf_node_ids:
            particle_regions[leaf_node_id]=[]

        #create list for number of regions each particle is in
        particle_region_count=np.zeros((node_count))

        #fill both arrays
        for leaf_node_id in leaf_node_ids:
            for particle_region_particle in region_particles[leaf_node_id]:
                particle_region_count[particle_region_particle]+=1

                particle_regions[particle_region_particle].append(leaf_node_id)

        #calculate weighted masses for each particle
        particle_weighted_masses=np.array(particle_masses)
        for leaf_node_id in leaf_node_ids:
            particle_weighted_masses[leaf_node_id]/=particle_region_count[leaf_node_id]

        #fill in sum tree for weighted masses
        fillFastSumTree(particle_weighted_masses,node_count,sum_order_summands,sum_order_targets)

        #helper functions for finding least number of nodes needed to calculate sums
        def rebuildWithParentNodes(leaf_nodes):
            tree=SubTree()
            for leaf_node in leaf_nodes:
                tree.add_node(nodes[leaf_node])
            return(List([node.array_indx for node in tree.nodes]))

        #create efficient sum trees and convert to numba lists
        optimized_region_particles = create_empty_nested_list(node_count)
        optimized_particle_regions = create_empty_nested_list(node_count)
        for leaf_node_id in leaf_node_ids:
            optimized_region_particles[leaf_node_id].extend(rebuildWithParentNodes(region_particles[leaf_node_id]))
            optimized_particle_regions[leaf_node_id].extend(rebuildWithParentNodes(particle_regions[leaf_node_id]))

        #use the efficent sum tree to find mass for each region
        region_weighted_masses = np.empty((node_count),dtype=np.float)
        useSumTree(particle_weighted_masses,node_count,optimized_region_particles,region_weighted_masses)

        #calculate weighted positions for each particle (for region center-of-mass)
        particle_weighted_positions=np.array(particle_positions)
        for leaf_node_id in leaf_node_ids:
            particle_weighted_positions[leaf_node_id]*=particle_weighted_masses[leaf_node_id]

        #fill in sum tree for weighted positions
        fillFastSumTree(particle_weighted_positions,node_count,sum_order_summands,sum_order_targets)

        #create array for region center-of-mass
        region_weighted_centers = np.empty((node_count,3),dtype=np.float)

        #use sum tree to quickly add up weighted positions for each region
        useSumTreeMatrix3(particle_weighted_positions,node_count,optimized_region_particles,region_weighted_centers)

        #divide by total mass to find center-of-mass
        for leaf_node_id in leaf_node_ids:
            if(region_weighted_masses[leaf_node_id]==0):
                raise Exception("firisk")
            region_weighted_centers[leaf_node_id]/=region_weighted_masses[leaf_node_id]

        #copy centers to initial centers array
        region_initial_weighted_centers = np.array(region_weighted_centers)

        #create arrays for future use
        particle_rotation_prereqs=np.zeros((node_count,3,3))
        region_rotation_prereqs=np.zeros((node_count,3,3))
        region_transforms=np.zeros((node_count,3,4))
        particle_transforms=np.zeros((node_count,3,4))

        particle_velocities=np.zeros((node_count,3))

        particle_targets=np.zeros((node_count,3))

        return (nodes, node_count, sum_order_summands, sum_order_targets, leaf_nodes, leaf_node_ids, particle_initial_positions, particle_positions, particle_masses, region_particles, particle_regions, particle_region_count, particle_weighted_masses, optimized_region_particles, optimized_particle_regions, region_weighted_masses, particle_weighted_positions, region_weighted_centers, region_initial_weighted_centers, particle_rotation_prereqs, region_rotation_prereqs, region_transforms, particle_transforms, particle_velocities,particle_targets)

@nb.njit(parallel=True)
def useSumTree(tree_input_array,node_count,to_sum,output_array):
    for i in nb.prange(node_count):
        curr_sum=0
        for j in range(len(to_sum[i])):
            curr_sum+=tree_input_array[to_sum[i][j]]
        output_array[i]=curr_sum

@nb.njit(parallel=True)
def useSumTreeMatrix3(tree_input_array,node_count,to_sum,output_array):
    for i in nb.prange(node_count):
        curr_sum=np.zeros((3))
        for j in range(len(to_sum[i])):
            curr_sum+=tree_input_array[to_sum[i][j]]
        output_array[i]=curr_sum

@nb.njit(parallel=True)
def useSumTreeMatrix3x3(tree_input_array,node_count,to_sum,output_array):
    for i in nb.prange(node_count):
        curr_sum=np.zeros((3,3))
        for j in range(len(to_sum[i])):
            curr_sum+=tree_input_array[to_sum[i][j]]
        output_array[i]=curr_sum


@nb.njit(parallel=True)
def useSumTreeMatrix3x4(tree_input_array,node_count,to_sum,output_array):
    for i in nb.prange(node_count):
        curr_sum=np.zeros((3,4))
        for j in range(len(to_sum[i])):
            curr_sum+=tree_input_array[to_sum[i][j]]
        output_array[i]=curr_sum


def fillFastSumTree(target_array,node_count,sum_order_summands,sum_order_targets):
    for i in range(len(sum_order_summands)):
        node_sum=0
        for j in range(8):
            if(sum_order_summands[i][j] == -1):
                continue
            node_sum+=target_array[sum_order_summands[i][j]]
        target_array[sum_order_targets[i]]=node_sum

import timeit

debug_time=None
def msr_time(sstr=None):
    global debug_time
    if(sstr is None):
        print("----")
        debug_time=timeit.default_timer()
    else:
        now=timeit.default_timer()
        print(sstr+": "+str((now-debug_time)*1000)+"ms")
        debug_time=now

@nb.jit(parallel=True)
def _step_calculate_weighted_positions(leaf_node_ids,leaf_node_count,particle_weighted_positions,particle_weighted_masses):
    for leaf_node_id_indx in nb.prange(leaf_node_count):
        particle_weighted_positions[leaf_node_ids[leaf_node_id_indx]]*=particle_weighted_masses[leaf_node_ids[leaf_node_id_indx]]

@nb.jit(parallel=True)
def _step_divide_by_total_mass(leaf_node_ids,leaf_node_count,region_weighted_centers,region_weighted_masses):
    #divide by total mass to find center-of-mass
    for leaf_node_id_indx in nb.prange(leaf_node_count):
        region_weighted_centers[leaf_node_ids[leaf_node_id_indx]]/=region_weighted_masses[leaf_node_ids[leaf_node_id_indx]]

@nb.jit(parallel=True)
def _step_calculate_rotation_prereqs(leaf_node_ids,leaf_node_count,particle_rotation_prereqs,particle_weighted_masses,particle_positions,particle_initial_positions):
    #calculate rotation prereqs for each particle (for region rotation)
    for leaf_node_id_indx in nb.prange(leaf_node_count):
        leaf_node_id=leaf_node_ids[leaf_node_id_indx]
        particle_rotation_prereqs[leaf_node_id]=particle_weighted_masses[leaf_node_id]*(np.expand_dims(particle_positions[leaf_node_id],1)@np.expand_dims(particle_initial_positions[leaf_node_id],0))

@nb.jit(parallel=True)
def _step_substract_per_region_constant(leaf_node_ids,leaf_node_count,region_rotation_prereqs,region_weighted_masses,region_weighted_centers,region_initial_weighted_centers):
    #substract per region constant for rotation prereqs
    for leaf_node_id_indx in nb.prange(leaf_node_count):
        leaf_node_id=leaf_node_ids[leaf_node_id_indx]
        region_rotation_prereqs[leaf_node_id]-=region_weighted_masses[leaf_node_id]*(np.expand_dims(region_weighted_centers[leaf_node_id],1)@np.expand_dims(region_initial_weighted_centers[leaf_node_id],0))

@nb.jit(parallel=True)
def _step_divide_transform_by_number_of_regions(leaf_node_ids,leaf_node_count,particle_transforms,particle_region_count):
    #divide transform by number of regions
    for leaf_node_id_indx in nb.prange(leaf_node_count):
        particle_transforms[leaf_node_ids[leaf_node_id_indx]]/=particle_region_count[leaf_node_ids[leaf_node_id_indx]]

@nb.jit(parallel=True)
def _step_calculate_particle_target(leaf_node_ids,leaf_node_count,particle_targets,particle_transforms,particle_initial_positions):
    #calculate particle target from transform
    for leaf_node_id_indx in nb.prange(leaf_node_count):
        leaf_node_id=leaf_node_ids[leaf_node_id_indx]
        particle_targets[leaf_node_id]=(particle_transforms[leaf_node_id]@np.append(particle_initial_positions[leaf_node_id],1))

def step(nodes, node_count, sum_order_summands, sum_order_targets, leaf_nodes, leaf_node_ids, particle_initial_positions, particle_positions, particle_masses, region_particles, particle_regions, particle_region_count, particle_weighted_masses, optimized_region_particles, optimized_particle_regions, region_weighted_masses, particle_weighted_positions, region_weighted_centers, region_initial_weighted_centers, particle_rotation_prereqs, region_rotation_prereqs, region_transforms, particle_transforms, particle_velocities,particle_targets):
    leaf_node_count=len(leaf_nodes)
    #calculate weighted positions for each particle (for region center-of-mass)
    np.copyto(particle_weighted_positions,particle_positions)
    _step_calculate_weighted_positions(leaf_node_ids,leaf_node_count,particle_weighted_positions,particle_weighted_masses)
    #fill in sum tree for weighted positions
    fillFastSumTree(particle_weighted_positions,node_count,sum_order_summands,sum_order_targets)

    #use sum tree to quickly add up weighted positions for each region
    useSumTreeMatrix3(particle_weighted_positions,node_count,optimized_region_particles,region_weighted_centers)

    _step_divide_by_total_mass(leaf_node_ids,leaf_node_count,region_weighted_centers,region_weighted_masses)

    _step_calculate_rotation_prereqs(leaf_node_ids,leaf_node_count,particle_rotation_prereqs,particle_weighted_masses,particle_positions,particle_initial_positions)

    #fill in sum tree for rotation prereqs
    fillFastSumTree(particle_rotation_prereqs,node_count,sum_order_summands,sum_order_targets)

    #use sum tree to quickly add up rotation prereqs for each region
    useSumTreeMatrix3x3(particle_rotation_prereqs,node_count,optimized_region_particles,region_rotation_prereqs)

    _step_substract_per_region_constant(leaf_node_ids,leaf_node_count,region_rotation_prereqs,region_weighted_masses,region_weighted_centers,region_initial_weighted_centers)

    #calculate region transforms
    for leaf_node_id in leaf_node_ids:
        u, p =polar(region_rotation_prereqs[leaf_node_id],side="left")
        region_transforms[leaf_node_id]=np.concatenate((u,np.expand_dims(region_weighted_centers[leaf_node_id]-np.matmul(u,region_initial_weighted_centers[leaf_node_id]),1)),axis=1)

    #fill in sum tree for region transforms
    fillFastSumTree(region_transforms,node_count,sum_order_summands,sum_order_targets)

    #use sum tree to calculate per-particle transforms
    useSumTreeMatrix3x4(region_transforms,node_count,optimized_particle_regions,particle_transforms)

    _step_divide_transform_by_number_of_regions(leaf_node_ids,leaf_node_count,particle_transforms,particle_region_count)

    _step_calculate_particle_target(leaf_node_ids,leaf_node_count,particle_targets,particle_transforms,particle_initial_positions)

class SubTree:
    def __init__(self,nodes=None):
        self.nodes=set()
        self.nodeParents={}
        if nodes is not None:
            for node in nodes:
                self.add_node(node)
    def add_node(self,node):
        if node.parent is None:
            self.nodes.add(node)
            return
        if not (node.parent in self.nodeParents):
            self.nodeParents[node.parent]=[]

        self.nodeParents[node.parent].append(node)
        self.nodes.add(node)

        if(len(self.nodeParents[node.parent])>=8):
            for node in self.nodeParents[node.parent]:
                self.nodes.remove(node)
            del self.nodeParents[node.parent]
            self.add_node(node.parent)
    def get_leaf_nodes(self):
        res=[]
        for node in self.nodes:
            res.extend(node.get_leaf_nodes())
        return(res)

def test_lattice_builder_node():
    lbn=LatticeBuilderNode()
    def frange(x, y, jump):
      while x < y:
        yield x
        x += jump

    def jk(xx,yy,zz):
        for x in frange(.47+xx,.53+xx,2**-6):
            for y in frange(.4+yy,.7+yy,2**-6):
                for z in frange(.47+zz,.53+zz,2**-6):
                    lbn.set(np.array((x,y,z)),5,True)
    jk(.11,0,.11)
    jk(-.11,0,.11)
    jk(-.11,0,-.11)
    jk(.11,0,-.11)

    for x in frange(.4,.61,2**-6):
        for y in frange(.5,1 ,2**-6):
            for z in frange(.4,.61,2**-6):
                lbn.set(np.array((x,y,z)),4,True)


    nodes, node_count, sum_order_summands, sum_order_targets, leaf_nodes, leaf_node_ids, particle_initial_positions, particle_positions, particle_masses, region_particles, particle_regions, particle_region_count, particle_weighted_masses, optimized_region_particles, optimized_particle_regions, region_weighted_masses, particle_weighted_positions, region_weighted_centers, region_initial_weighted_centers, particle_rotation_prereqs, region_rotation_prereqs, region_transforms, particle_transforms, particle_velocities, particle_targets = lbn.buildParticleArrays(np.zeros(3),np.ones(3),np.ones(3)*.15,(lambda c,g:g))

    time_step=.1;
    gravity = np.array([0,-.1,0])
    points=[(.2,.4,.4),(.6,.6,.6)]
    debug_points=[(0,0,0),(0,1,0)]

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color((0,0,0))
    vis.add_geometry(pcd)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(debug_points)
    pcd2.paint_uniform_color((0,1,0))
    vis.add_geometry(pcd2)

    while True:

        step(nodes, node_count, sum_order_summands, sum_order_targets, leaf_nodes, leaf_node_ids, particle_initial_positions, particle_positions, particle_masses, region_particles, particle_regions, particle_region_count, particle_weighted_masses, optimized_region_particles, optimized_particle_regions, region_weighted_masses, particle_weighted_positions, region_weighted_centers, region_initial_weighted_centers, particle_rotation_prereqs, region_rotation_prereqs, region_transforms, particle_transforms, particle_velocities,particle_targets)
        for leaf_node_id in leaf_node_ids:
            particle_velocities[leaf_node_id]+=time_step*gravity
            particle_velocities[leaf_node_id]*=.97
            particle_velocities[leaf_node_id]+=(particle_targets[leaf_node_id]-particle_positions[leaf_node_id])/time_step
            particle_positions[leaf_node_id]+=time_step*particle_velocities[leaf_node_id]

            if(particle_positions[leaf_node_id][1]<.2):
                particle_positions[leaf_node_id][1]=.2
                particle_velocities[leaf_node_id][1]=0
        points=[]
        debug_points=[]
        for leaf_node_id in leaf_node_ids:
            points.append(particle_positions[leaf_node_id])
            #debug_points.append(particle_targets[leaf_node_id])


        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color((0,0,0))
        pcd2.points = o3d.utility.Vector3dVector(debug_points)
        pcd2.paint_uniform_color((0,1,0))

        vis.update_geometry(pcd)
        vis.update_geometry(pcd2)

        vis.poll_events()
        vis.update_renderer()
if __name__ == "__main__":
    test_lattice_builder_node()
