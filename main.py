
def lerp(a,b,p):
    return(a*(1-p)+b*p)
import numpy as np
import open3d as o3d
import time

from scipy.linalg import polar
import numba as nb

class OctTreeNode():
    def __init__(self,mass=1,is_leaf=False,parent=None):
        self.children=np.full((2,2,2), None, dtype=np.object)
        self.is_leaf=is_leaf
        self.parent=parent
        self.mass=mass
        self.region_count=0
    '''
    special behavior:
    if setting filled and target is too coarse, new nodes are created, even if it is already filled:

    |   |* *|    |   |*|*|
    |-------| -> |-------|
    |   |   |    |   |   |

    This does not apply to filling empty because particles are only made on filled ones
    '''

    def make_children(self,filled):
        if not self.is_leaf:
            raise Exception("Cannot fill children because not leaf")
        self.is_leaf=False

        if(filled):
            for x in range(2):
                for y in range(2):
                    for z in range(2):
                        self.children[(x,y,z)]=OctTreeNode(is_leaf=True,parent=self)

        else:
            self.children=np.full((2,2,2),None, dtype=np.object)

    def set(self,point, depth, filled): # -> is_empty (only if filled = false)
        if depth<0:
            raise Exception("invalid depth")
        if depth==0:
            raise Exception("cannot set current node (need parent)")
        target_node_key=tuple(np.array(point*2,dtype=np.int_))
        target_node_pos=(point*2)%1
        if filled:
            if self.is_leaf:
                self.make_children(True)
            if depth==1:
                self.children[target_node_key]=OctTreeNode(is_leaf=True,parent=self)
            else:
                if(self.children[target_node_key] is None):
                    self.children[target_node_key]=OctTreeNode(is_leaf=True,parent=self)
                    self.children[target_node_key].make_children(False)
                self.children[target_node_key].set(target_node_pos,depth-1,filled)
        else:
            if self.is_leaf:
                raise Exception("cannot delete leaf (need parent)")
            if depth==1:
                self.children[target_node_key]=None
            else:
                if(self.children[target_node_key] is not None):
                    if self.children[target_node_key].is_leaf:
                        self.children[target_node_key].make_children(True)

                    if(self.children[target_node_key].set(target_node_pos,depth-1,filled)):
                        self.children[target_node_key]=None
            return(np.all(self.children==None))

    def display(self):
        pcd = o3d.geometry.PointCloud()
        points=self.get_points()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color((0,0,0))

        pcd2 = o3d.geometry.PointCloud()
        points2=self.get_emptys()
        pcd2.points = o3d.utility.Vector3dVector(points2)
        pcd2.paint_uniform_color((1,0,0))
        o3d.visualization.draw_geometries([pcd2,pcd])

    def middle(lower,upper):
        return((lower+upper)/2)
    def get_initial_points(self):
        if(self.is_leaf):
            return([self.initial_position])
        res=[]
        for x in range(2):
            for y in range(2):
                for z in range(2):
                    if(self.children[(x,y,z)] is None):
                        continue
                    res.extend(self.children[(x,y,z)].get_initial_points())
        return(res)
    def get_points(self):
        if(self.is_leaf):
            return([self.position])
        res=[]
        for x in range(2):
            for y in range(2):
                for z in range(2):
                    if(self.children[(x,y,z)] is None):
                        continue
                    res.extend(self.children[(x,y,z)].get_points())
        return(res)

    def set_positions(self,lower,upper):
        if(self.is_leaf):
            self.initial_position=OctTreeNode.middle(lower,upper)
            self.position=np.array(self.initial_position)
            self.velocity = np.zeros((3))
            return
        self.initial_position_lower=lower
        self.initial_position_upper=upper
        md=[lower,OctTreeNode.middle(lower,upper),upper]
        for x in range(2):
            for y in range(2):
                for z in range(2):
                    if(self.children[(x,y,z)] is None):
                        continue
                    self.children[(x,y,z)].set_positions(
                        np.array((md[x][0],md[y][1],md[z][2])),
                        np.array((md[x+1][0],md[y+1][1],md[z+1][2]))
                    )

    def get_emptys(self,lower,upper):
        if(self.is_leaf):
            return([])
        middles=[self.initial_position_lower,OctTreeNode.middle(self.initial_position_lower,self.initial_position_upper),self.initial_position_upper]
        print(middles)
        res=[]
        for x in range(2):
            for y in range(2):
                for z in range(2):
                    if(self.children[(x,y,z)] is None):
                        res.append(OctTreeNode.middle(
                            np.array((  middles[x][0],   middles[y][1],   middles[z][2])),
                            np.array((middles[x+1][0], middles[y+1][1], middles[z+1][2]))
                        ))
                        continue
                    res=res+(self.children[(x,y,z)].get_emptys(
                        np.array((  middles[x][0],   middles[y][1],   middles[z][2])),
                        np.array((middles[x+1][0], middles[y+1][1], middles[z+1][2]))
                    ))
        return(res)

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

    def _get_AABB_nodes(self,lower,upper):
        middles=[self.initial_position_lower,OctTreeNode.middle(self.initial_position_lower,self.initial_position_upper),self.initial_position_upper]

        res=[]
        for x in range(2):
            xo=OctTreeNode.overlap(lower[0],upper[0],middles[x][0],middles[x+1][0])
            for y in range(2):
                yo=OctTreeNode.overlap(lower[1],upper[1],middles[y][1],middles[y+1][1])
                for z in range(2):
                    zo=OctTreeNode.overlap(lower[2],upper[2],middles[z][2],middles[z+1][2])
                    o=min(xo,yo,zo)
                    if(o==0):
                        pass
                    elif(o==1):
                        if self.children[(x,y,z)] is not None:
                            if(self.children[(x,y,z)].is_leaf):
                                res.append(self.children[(x,y,z)])
                            else:
                                res=res+self.children[(x,y,z)]._get_AABB_nodes(lower,upper)
                    elif(o==2):
                        if self.children[(x,y,z)] is not None:
                            res.append(self.children[(x,y,z)])
        return res;
    def get_AABB_nodes(self,lower,upper):
        return(SubTree(nodes=self._get_AABB_nodes(lower,upper)))
    def get_leaf_nodes(self):
        if(self.is_leaf):
            return([self])
        res=[]
        for x in range(2):
            for y in range(2):
                for z in range(2):
                    if(self.children[(x,y,z)] is not None):
                        res=res+self.children[(x,y,z)].get_leaf_nodes()
        return(res)
    def set_sums(self,attr):
        if(self.is_leaf):
            return(getattr(self,attr))
        sum=0
        for x in range(2):
            for y in range(2):
                for z in range(2):
                    if(self.children[(x,y,z)] is not None):
                        sum+=self.children[(x,y,z)].set_sums(attr)
        setattr(self,attr,sum)
        return(sum)
    def calculate_transform_prereqs(self):
        self.weighted_position = self.weighted_mass*self.position
        self.rotation_prereq = self.weighted_mass*(np.expand_dims(self.position,1)@np.expand_dims(self.initial_position,0))
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


def frange(x, y, jump):
  while x < y:
    yield x
    x += jump


class PhysRegion:
    def __init__(self,region):
        self.region=region
    def calculate_mass_info(self):
        total_mass=0
        center_of_mass=np.zeros((3))
        for node in self.region.get_leaf_nodes():
            total_mass+=node.weighted_mass
            center_of_mass+=node.position*node.weighted_mass
        center_of_mass/=total_mass
        return(total_mass,center_of_mass)

    def calculate_transform(self):
        center_of_mass=np.zeros((3))
        rotation_prereq_sum=np.zeros((3,3))

        for node in self.region.nodes:
            center_of_mass+=node.weighted_position
            rotation_prereq_sum+=node.rotation_prereq

        #print(str(len(self.region.nodes))+"/"+str(len(self.region.get_leaf_nodes())))
        center_of_mass/=self.total_mass
        rotation_prereq_sum-=self.total_mass*(np.expand_dims(center_of_mass,1)@np.expand_dims(self.i_center_of_mass,0))

        u, p =polar(rotation_prereq_sum,side="left")

        self.transform=np.concatenate((u,np.expand_dims(center_of_mass-np.matmul(u,self.i_center_of_mass),1)),axis=1)

    def update_particle_transforms(self,time_step):

        for node in self.region.get_leaf_nodes():

            node.transform+=self.transform
            node.transform_count+=1
def test_physics():
    otn=OctTreeNode()
    '''
    for x in frange(.4,.6,2**-5):
        for y in frange(.4,.6,2**-5):
            for z in frange(.4,.6,2**-5):
                if(np.linalg.norm(np.array((x,y,z))-np.array((.5,.5,.5)))<.1):
                    otn.set(np.array((x,y,z)),5,True)


    for x in frange(.6,.8,2**-6):
        for y in frange(.34,.58,2**-6):
            for z in frange(.42,.58,2**-6):
                otn.set(np.array((x,y,z)),4,True)
    '''
    def jk(xx,yy,zz):
        for x in frange(.47+xx,.53+xx,2**-6):
            for y in frange(.4+yy,.7+yy,2**-6):
                for z in frange(.47+zz,.53+zz,2**-6):
                    otn.set(np.array((x,y,z)),4,True)
    jk(.11,0,.11)
    jk(-.11,0,.11)
    jk(-.11,0,-.11)
    jk(.11,0,-.11)

    for x in frange(.4,.61,2**-6):
        for y in frange(.6,.7 ,2**-6):
            for z in frange(.4,.61,2**-6):
                otn.set(np.array((x,y,z)),4,True)

    otn.set_positions(np.zeros(3),np.ones(3))
    points=otn.get_points()
    leaf_nodes=otn.get_leaf_nodes()

    regions=[]
    for point in points:
        region_subtree=otn.get_AABB_nodes(point-.12,point+.12)

        regions.append(PhysRegion(region_subtree))
        for node in region_subtree.get_leaf_nodes():
            node.region_count+=1

    for node in leaf_nodes:
        node.weighted_mass=node.mass/max(node.region_count,1)
    for region in regions:
        region.total_mass,region.i_center_of_mass=region.calculate_mass_info()



    vis = o3d.visualization.Visualizer()
    vis.create_window()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(otn.get_points())
    pcd.paint_uniform_color((0,0,0))
    vis.add_geometry(pcd)

    time_step=.1
    gravity=np.array((.0,-.1,0))

    debug_points=[]
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(debug_points)
    pcd2.paint_uniform_color((0,0,0))
    vis.add_geometry(pcd2)

    mesh_box = o3d.geometry.TriangleMesh.create_box(width=1.0,
                                                height=.0005,
                                                depth=1.0)
    mesh_box=mesh_box.translate((0, .2, 0))
    vis.add_geometry(mesh_box)

    _dbg_indx=0
    dbg_indx=0
    while True:
        _dbg_indx+=.02
        dbg_indx=int(_dbg_indx)

        debug_points=[]
        debug_points_colors=[]
        for node in leaf_nodes:
            node.calculate_transform_prereqs()
        otn.set_sums("weighted_position")
        otn.set_sums("rotation_prereq")

        for node in leaf_nodes:
            node.calculate_transform_prereqs()

        for region in regions:
            region.calculate_transform()

        for node in leaf_nodes:
            node.transform_count=0
            node.transform=np.zeros((3,4))

        for region in regions:
            region.update_particle_transforms(time_step)

        for node in leaf_nodes:

            target_pos=np.matmul(node.transform/max(1,node.transform_count),np.append(node.initial_position,1))
            #debug_points.append(target_pos)
            node.velocity+=(target_pos-node.position)/time_step
            #debug_points_colors.append([0,1,0])

        for node in leaf_nodes:
            node.velocity+=time_step*gravity
            node.position+=time_step*node.velocity
            node.velocity*=.95
        for node in leaf_nodes:
            if(node.position[1]<.2):
                node.position[1]=.2
                node.velocity[1]=0

        for particle in regions[dbg_indx].region.get_leaf_nodes():
            break
            debug_points.append(particle.position)
            debug_points_colors.append([0,1,0])
            print(particle.initial_position)
            debug_points.append(particle.initial_position)
            debug_points_colors.append([0,0,1])

        pcd.paint_uniform_color((1,0,0))
        pcd.points = o3d.utility.Vector3dVector(otn.get_points())
        vis.update_geometry(pcd)

        pcd2.paint_uniform_color((0,1,0))
        pcd2.points = o3d.utility.Vector3dVector(debug_points)
        pcd2.colors = o3d.utility.Vector3dVector(debug_points_colors)

        vis.update_geometry(pcd2)

        vis.poll_events()
        vis.update_renderer()
test_physics()














def test_octTree():
    otn=OctTreeNode()
    '''
    for x in frange(.2,.6,2**-5):
        for y in frange(.4,.6,2**-5):
            for z in frange(.4,.6,2**-5):
                otn.set(np.array((x,y,z)),4,True)
    '''

    otn.set(np.array((.5,.51,.49)),3,True)
    otn.set(np.array((.5,.51,.51)),3,True)
    otn.set(np.array((.5,.49,.51)),3,True)
    otn.set(np.array((.5,.49,.49)),3,True)

    otn.set_positions(np.zeros(3),np.ones(3))
    points=otn.get_points()


    vis = o3d.visualization.Visualizer()
    vis.create_window()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color((0,0,0))
    vis.add_geometry(pcd)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector([])
    vis.add_geometry(pcd2)
    indx=0
    _indx=0

    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=.01)
    mesh_sphere.compute_vertex_normals()
    mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])
    vis.add_geometry(mesh_sphere)

    prevpoint=np.array([0,0,0])

    #   lower       upper
    #   |----------|       (2)
    #      [-----]
    #      a    b

    while True:
        _indx+=.1
        indx=int(_indx)

        subtree=otn.get_AABB_nodes(points[indx]-.1,points[indx]+.1)


        positions=[]
        for leaf_node in subtree.get_leaf_nodes():
            positions.append(leaf_node.initial_position)


        pcd2.paint_uniform_color((1,0,0))
        pcd2.points = o3d.utility.Vector3dVector(positions)
        vis.update_geometry(pcd2)


        mesh_sphere.translate(np.array(points[indx])-prevpoint)
        prevpoint=np.array(points[indx])
        vis.update_geometry(mesh_sphere)

        vis.poll_events()
        vis.update_renderer()
        time.sleep(.01)



    '''
    def definePositions(min,max):
        self.pos_min=min
        self.pos_max=max
        x_poses=[lerp(min[0],max[0],0),lerp(min[0],max[0],1/3),lerp(min[0],max[0],2/3),lerp(min[0],max[0],1)]
        y_poses=[lerp(min[1],max[1],0),lerp(min[1],max[1],1/3),lerp(min[1],max[1],2/3),lerp(min[1],max[1],1)]
        z_poses=[lerp(min[2],max[2],0),lerp(min[2],max[2],1/3),lerp(min[2],max[2],2/3),lerp(min[2],max[2],1)]
        for x in range(3):
            for y in range(3):
                for z in range(3):
                    if(self.children[x][y][z] is not None):
                        self.children[x][y][z].definePositions((x_poses[x],y_poses[y],z_poses[z]),(x_poses[x+1],y_poses[y+1],z_poses[z+1]))
    '''


    '''
    class OctTree():
        def __init__(self, baseGridSize):
            self.baseGrid={}
            self.baseGridSize=baseGridSize
        def build(buildfunc,initial):#buildfunc(position) -> (filled,max_resolution,min_resolution)
    '''
