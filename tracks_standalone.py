import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#John Van Atta
#12 June 2013

#3D Hexagonal Pattern Recognition
#Based on NIFFTE experimental setup


class Voxel(object):
    """
    Voxels are the basic data unit. They are a sort of 3d hexagonal pixel, using two spatial and one temporal coordinate.
    The first value on the IDarray is a chamber identifier, not used much
    The second two values are spatial coordinates, the third time position, used as another coordinate
    The adc value is the magnitude of the signal recorded
    """
    
    def __init__(self, idarray, adc):
        self.id = (idarray[0], idarray[1], idarray[2], idarray[3])
        self.adc = adc
        
    def getID(self):
        return self.id
    
    def getVal(self):
        return self.adc
    
    def toString(self):
        return str(self.id) + "  " + str(self.adc)


class Event(object):
    """
    Events start with all the voxels for an event. The voxels are processed using makeTrajectories() into Trajectory objects
    The Event class also contains methods to merge multiple trajectories with similar directions and prune small, spurious trajectories    
    """
    
    def __init__(self, id):
        self.id = id
        self.edat = []
        self.traj = []
        self.orphans = []
                                       
    def getID(self):
        return self.id
    
    def getData(self):
        return self.edat
                    
    def getTrajectories(self):
        return self.traj
    
    def getOrphans(self):
        return self.orphans
            
    def printData(self):
        for v in self.edat:
            print v.toString()
    
    def addVoxel(self, d):
        self.edat.append(Voxel([int(d[0]), int(d[1]), int(d[2]), int(d[3])] , int(d[4])))
    

    def popNeighbors(self, vox):
        """
        Locate any neighbors of a given voxel in the event's list.
        Neighbors are removed from the event's list--make sure they get put somewhere.        
        """
        poplist = []
        for v in listNeighbors(self.edat, vox):
            poplist.append(self.edat.pop(self.edat.index(v)))
        return poplist
    
    
    def makeTrajectories(self, gradthresh, dirthresh):
        """
        Convert an event's voxels into one or more trajectories. Takes in two threshold values.
        First, the voxels are sorted by signal strength. The largest is chosen to start a trajectory.
        Next, its neighbors are found. The neighbor with the smallest gradient (from the current) is added to the spine of the trajectory
        Other neighbors are checked. If they are under thresholds, they are added to the flesh of the trajectory. Checking here creates some orphans,
        it's likely that the code could be improved by only checking for spine members and just adding all their neighbors.
        When no neighbors satisfy the requirements, unused voxels are recycled back into the event list and a new trajectory is created.
        Eventually only 1 length trajectories are created from the residual voxels. These turned into the event's orphans.
        
        In: gradthresh is a multiplied by the voxel's signal strength as a relative weight.
        dirthresh is an absolute threshold. The direction testing method returns the sum of the squared differences, which are compared
        against dirthresh.
        """
        #sort based on voxel value
        self.edat = sorted(self.edat, key = lambda v: v.getVal(), reverse=True)
        
        maxlen = len(self.edat)
        
        #eventually adds everything to a trajectory        
        while len(self.edat) > 0:
            
            #start a new trajectory
            vox = self.edat.pop(0)
            newTraj = Trajectory(vox)
            self.traj.append(newTraj)
            
            #breaks when there are no more candidates being generated
            while True:
                
                candidates = []
                rejects = []
                chosen = []
                
                #add more voxels from the neighbors
                neigh = self.popNeighbors(vox)
                candidates.extend(neigh)
                
                grad = getGradient(vox, neigh)
                
                #check which voxels satisfying gradient and direction requirements
                for v in candidates:
                    thresh = vox.getVal() * gradthresh
                    if grad[v][1] <= thresh:
                        if newTraj.checkDir(grad[v][0], dirthresh):
                            #voxel matches trajectory
                            chosen.append(v)
                        else:
                            rejects.append(v)
                    else:
                        rejects.append(v)
                
                #no new chosen, time to break out of the loop
                if len(chosen) == 0:
                    self.edat.extend(rejects)
                    break
                
                #find the lowest gradient of the neighbors. that will be our new tail
                mininum = 1000000
                lowestgrad = None
                for v in chosen:
                    if grad[v][1] < mininum:
                        mininum = grad[v][1]
                        lowestgrad = v
                vox = chosen.pop(chosen.index(lowestgrad))
                newTraj.addSpine(vox)
                
                #add the rest of the voxels that match gradient and direction but are not best
                for v in chosen:
                    newTraj.addFlesh(v)
                #recycle unused voxels            
                self.edat.extend(rejects)
            #end of chained tail loop
            
            #resort
            self.edat = sorted(self.edat, key = lambda v: v.getVal(), reverse=True)
            
        #Unpaired voxels end as length 1 trajectories. Destroy these and add the voxels to event's orphans.
        for t in reversed(self.traj):
            if len(t.getSpine()) == 1:
                self.orphans.extend(t.getMembers())
                self.traj.remove(t)
        
    def mergeTrajectories(self, mergethresh):
        """
        Combine multiple trajectories with similar directions and near-matching endpoints into one trajectory.
        Checks each trajectory against all the other trajectories in an event
        If the direction is close, determined by mergethresh, AND they have a near-matched endpoint, they are merged. Directions are unsigned,
        moving top-to-bottom is considered the same as moving bottom-to-top
        The loop keeps trying to merge until it has done an iteration with no merges in case merging one trajectory allows a second to be merged.
        The trajectory endpoints dont have to be neighbors. They are allowed a small separation. This solves two problems. First, the situation
        where both endpoints have added all their neighbors to their trajectory, leaving a one-voxel gap between spine members.
        Second, an experimentally defective voxel abruptly ending a trajectory (flawed data). Unfortunately, it could introduce an error
        where two true trajectories with similar directions and translated-but-close endpoints are incorrectly merged. This sort of occurence does
        not appear to be common in the data.
        
        In: mergethresh is an absolute parameter similar to dirthresh in makeTrajectories.
        """
        counter = 0
        
        #breaks after no merge activity has occured
        loop = True
        while loop:
            loop = False
            numtraj = len(self.traj)
            #for every trajectory, compare against all other trajectories in event
            for x in range(numtraj):
                t2 = self.traj.pop(0)
                tdir = t2.getDir()
                merged = False
                for i in range(numtraj):
                    if i >= len(self.traj):
                        break
                    t = self.traj[i]
                    #check the unsigned direction agains the threshold, AND check if either trajectory's head or tail are closely located
                    if t.checkDirReversible(tdir, mergethresh) and (voxDistance(t.getHead(), t2.getHead()) < 10 or voxDistance(t.getTail(), t2.getHead()) < 10 or voxDistance(t.getHead(), t2.getTail()) < 10 or voxDistance(t.getTail(), t2.getTail()) < 10):
                            
                        t.merge(t2)
                        merged = True
                        loop = True
                        counter += 1
                        break
                if not merged:
                    self.traj.append(t2)
        return counter
        
    def cleanTrajectories(self, cleanthresh):
        """
        Destroy small trajectories with total size less than cleanthresh. They are added to orphans.
        """
        counter = 0
        for t in reversed(self.traj):
            if len(t.getMembers()) <= cleanthresh:
                self.orphans.extend(t.getMembers())
                self.traj.remove(t)
                counter += 1
        return counter


class Trajectory(object):
    """
    Trajectories hold processed voxels in a list. Voxels are broken into two categories: spine voxels and flesh voxels
    Spine voxels dictate the trajectory's direction. Flesh voxels have no effect on the trajectory-the trajectory is just used to store them.
    When a new spine member is added, the trajectory's direction is updated.
    Currently there is no weighting on the direction based on confidence.    
    """
    def __init__(self, vox):
        self.du = 0.
        self.dv = 0.
        self.dw = 0.
        self.directions = []
        self.flesh = []
        self.spine = [vox]
        self.tail = vox
    
    def addFlesh(self, vox):
        self.flesh.append(vox)
        
    def addSpine(self, vox):
        """
        Add a voxel to the Trajectory's spine. Direction is updated for every spine member added.
        """
        self.spine.append(vox)
        self.tail = vox
        self.directions.append(getGradient(self.spine[-2], [self.spine[-1]])[self.spine[-1]][0])
        self.du = sum([v[0] for v in self.directions]) / float(len(self.directions))
        self.dv = sum([v[1] for v in self.directions]) / float(len(self.directions))
        self.dw = sum([v[2] for v in self.directions]) / float(len(self.directions))
    
    def merge(self, tra):
        """
        Merge this trajectory with another trajectory. First, merge spines, then add all flesh members.
        """
        for v in tra.getSpine():
            self.addSpine(v)
        for v in tra.getFlesh():
            self.addFlesh(v)

    def getDir(self):
        return self.du,self.dv,self.dw
        
    def getTail(self):
        return self.spine[-1]
    
    def getHead(self):
        return self.spine[0]
    
    def getFlesh(self):
        return self.flesh
    
    def getSpine(self):
        return self.spine
    
    def getMembers(self):
        """
        Returns all members, spine and flesh
        """
        all = []
        all.extend(self.flesh)
        all.extend(self.spine)
        return all
    
    def checkDir(self, (vdu, vdv, vdw), dirthresh):
        """
        Checks this Trajectory's direction (signed) against a given tuple assumed to be a direction against a threshold.
        Squares the differences between this direction and the input one, DOES NOT SQUARE ROOT. Returns boolean.
        """
        if self.du == 0 and self.dv == 0 and self.dw ==0:
            return True
        
        err = (self.du - vdu)**2 + (self.dv - vdv)**2 + (self.dw - vdw)**2
        if err > dirthresh:
            return False
        
        return True

    def checkDirReversible(self, (vdu, vdv, vdw), dirthresh):
        """
        Checks this Trajectory's direction (unsigned) against a given tuple assumed to be a direction against a threshold.
        Squares the differences between this direction and the input one, DOES NOT SQUARE ROOT. Returns boolean.
        """
        if self.du == 0 and self.dv == 0 and self.dw ==0:
            return True
        
        #Reverse the direction, then use the lesser of the errors
        errNormal = (self.du - vdu)**2 + (self.dv - vdv)**2 + (self.dw - vdw)**2
        vdu *= -1
        vdv *= -1
        vdw *= -1
        errReversed = (self.du - vdu)**2 + (self.dv - vdv)**2 + (self.dw - vdw)**2
        
        err = min(errNormal, errReversed)
        if err > dirthresh:
            return False
        return True
    
    def toString(self):
        s = "Trajectory \n"
        s += "Direction: " + str(self.du) + ", " + str(self.dv) + ", " + str(self.dw) + "\n"
        s += "Head:" + self.spine[0].toString() + " ; Tail: " + self.spine[-1].toString() + "\n"
        s += "Spine Length: " + str(len(self.spine)) + ", Body Count: " + str(len(self.flesh)) + "\n-----"
        return s
        
          
def listNeighbors(li, vox):
        """
        Neighboring voxels in 3D hexagonal geometry.
        Finds all neighbors of an input voxel in an input list.
        Since spatial coordinates are treated as Cartesian, some manipulation is necessary to get the proper hexagonal neighbors.
        Each entry in nmap dictionary is a small adjustment of the given voxel's coordinates that represents one of the 20 neighbors in space and time.
        Alternating rows have a slightly different set of neighbor mappings.
        """
        ne = []
        q = vox.getID()
        if q[1]%2 == 1:
            nmap = {
                (q[0], q[1], q[2]-1, q[3]), (q[0], q[1]-1, q[2], q[3]), (q[0], q[1]-1, q[2]+1, q[3]),
                (q[0], q[1], q[2]+1, q[3]), (q[0], q[1]+1, q[2]+1, q[3]), (q[0], q[1]+1, q[2], q[3]),
                
                (q[0], q[1], q[2], q[3]-1), (q[0], q[1], q[2]-1, q[3]-1), (q[0], q[1]-1, q[2], q[3]-1),
                (q[0], q[1]-1, q[2]+1, q[3]-1), (q[0], q[1], q[2]+1, q[3]-1), (q[0], q[1]+1, q[2]+1, q[3]-1),
                (q[0], q[1]+1, q[2], q[3]-1),
                
                (q[0], q[1], q[2], q[3]+1), (q[0], q[1], q[2]-1, q[3]+1), (q[0], q[1]-1, q[2], q[3]+1),
                (q[0], q[1]-1, q[2]+1, q[3]+1), (q[0], q[1], q[2]+1, q[3]+1), (q[0], q[1]+1, q[2]+1, q[3]+1),
                (q[0], q[1]+1, q[2], q[3]+1)
                }
        else:
            nmap = {
                (q[0], q[1], q[2]-1, q[3]), (q[0], q[1]-1, q[2], q[3]), (q[0], q[1]-1, q[2]-1, q[3]),
                (q[0], q[1], q[2]+1, q[3]), (q[0], q[1]+1, q[2]-1, q[3]), (q[0], q[1]+1, q[2], q[3]),
                
                (q[0], q[1], q[2], q[3]-1), (q[0], q[1], q[2]-1, q[3]-1), (q[0], q[1]-1, q[2], q[3]-1),
                (q[0], q[1]-1, q[2]-1, q[3]-1), (q[0], q[1], q[2]+1, q[3]-1), (q[0], q[1]+1, q[2]-1, q[3]-1),
                (q[0], q[1]+1, q[2], q[3]-1),
                
                (q[0], q[1], q[2], q[3]+1), (q[0], q[1], q[2]-1, q[3]+1), (q[0], q[1]-1, q[2], q[3]+1),
                (q[0], q[1]-1, q[2]-1, q[3]+1), (q[0], q[1], q[2]+1, q[3]+1), (q[0], q[1]+1, q[2]-1, q[3]+1),
                (q[0], q[1]+1, q[2], q[3]+1)
                }
        #Given data should not have any duplicate voxels. Could be disabled to improve runtime.
        for v in li:
            #Any neighbor will have one of the permutations of nmap
            if v.getID() in nmap:
                unique = True
                for n in ne:
                    if v.getID() == n.getID():
                        unique = False
                        print "**Error: Duplicate voxel**", v.getID()
                if unique:        
                    ne.append(v)    
        return ne
    
def neighborCheck(avox, bvox):
    """
    Check if two voxels are neighbors of each other
    """
    if len(listNeighbors([avox], bvox)) == 1:
        return True
    else: return False
    
def voxDistance(avox, bvox):
    """
    Find the distance between two voxels. Returns the absolute sum of any differences in their three coordinates. No weighting or fanciness.
    """
    aID = avox.getID()
    bID = bvox.getID()
    
    if aID[0] != bID[0]:
        return 1e10
    return abs(aID[1] - bID[1]) + abs(aID[2] - bID[2]) + abs(aID[3] - bID[3])

def getGradient(vox, neigh):
    """
    Creates a gradient dictionary for an input voxel and a list of its neighbors. Non-neighbor gradient is untested.
    Dictionary uses the neighbor voxel as keys with tuple values ((du,dv,dw), dE)
    """
    grad = dict()
    for v in neigh:
        dE = vox.getVal() - v.getVal()
        du = vox.getID()[1] - v.getID()[1]
        dv = vox.getID()[2] - v.getID()[2]
        dw = vox.getID()[3] - v.getID()[3]
        
        grad[v] = (du,dv,dw), dE
    return grad

def voxelsToArray(li):
    """
    Convert a list of voxels into a numpy array for plotting
    """
    le = len(li)
    earr = [np.zeros(le), np.zeros(le), np.zeros(le), np.zeros(le), np.zeros(le)]
    for i, v in enumerate(li):
        earr[0][i] = v.getID()[0]
        earr[1][i] = v.getID()[1]
        earr[2][i] = v.getID()[2]
        earr[3][i] = v.getID()[3]
        earr[4][i] = v.getVal()
    return earr
    
def plotEvent(ev, eventnum, trajcount, usedcount, orphancount, orphansOn, filename):
    """
    Makes a 3D plot of an event. Different trajectories are color coded, with optional orphans in gray.
    Stronger signal voxels have larger point blobs.
    """

    fig = plt.figure(1, figsize=(10,8))
    ax = fig.add_subplot(1,1,1, projection = '3d')
    
    #Hardcode colors of the first 8 trajectories. This should usually be enough
    colors = ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink"]
    for i, t in enumerate(ev.getTrajectories()):
        evda = voxelsToArray(t.getMembers())
        #if there are too many trajectories, prevent errors by giving them brown color
        if i > 7:
            tcolor = 'brown'
        else :
            tcolor = colors[i]
        
        p = ax.scatter(evda[1], evda[2], evda[3], s=2*evda[4], linewidth=0, color=tcolor)
    
    #Orphans are optional, switched from input
    if orphansOn:
        evda = voxelsToArray(ev.getOrphans())
        tcolor = 'gray'
        p = ax.scatter(evda[1], evda[2], evda[3], s=2*evda[4], linewidth=0, color=tcolor)
    
    plt.xlim(10, 50)
    plt.ylim(10, 50)
    ax.set_zlim3d(10,50)
    
    #The title holds all the pertinent run information. Soem of this could probably be derived from ev instead of forcefed. 
    ti = "Event " + str(eventnum) + "\n\n" + "Trajectories: " + str(trajcount) + "\n" + "Used Voxels: " + str(usedcount) + "\n" + "Orphans: " + str(orphancount)
    plt.title(ti, horizontalalignment='left', x=.1)

    ax.set_xlabel('Row')
    ax.set_ylabel("Column")
    ax.set_zlabel('Bucket')
    
    plt.savefig(filename)
    plt.close()
    

############################################
#Start of procedural code
#Setup and parse the data
evlist = []
evcounter = 0
datain = open('niffte_data.txt', 'r')

for i, line in enumerate (datain):
    #New events are denoted by # blocks. If one is encountered, start a new event.
    if line.startswith('#'):
        evlist.append(Event(evcounter))
        evcounter += 1
    #Otherwise, keep adding voxel data to the current event.
    else:
        s = np.fromstring(line, sep=" ")
        evlist[-1].addVoxel(s)
        
datain.close()

#######
#Parameters to set


gradthreshold = .75
dirthreshold = 2.05
mthresh = .75

prunethresh = .08

showOrphans = True

#some useful events for testing
#1 has good alternate high glitch
#12 is a useful single track
# 29 is a softball split
#28 for testing directionality, it seems to start bottom right and go up
#10 three way split
#70 is a showdown
#78 three way split
#66 tracks bleed

#known good parameters: .75 2.05 .75

#----------------
#batch information
totalvox = 0
totaltraj = 0
totalorphans = 0

#plot every event, 100 in all
for i, ev in enumerate(evlist):
    
    run = ev

    oldlen = len(run.getData())
    #make the trajectories
    run.makeTrajectories(gradthreshold, dirthreshold)
    #merge trajectories
    nummerged = run.mergeTrajectories(mthresh)
    #prune out small trajectories as a percent of the original voxel count. More original voxels means more voxels have to be present to keep a trajectory.
    numpruned = run.cleanTrajectories(oldlen * prunethresh)
    alltraj = run.getTrajectories()
    
    #trajectory information
    usedlen = 0
    for t in alltraj:
        #detailed trajectory information
        #print t.toString()
        usedlen += len(t.getMembers())

    
    #run information
    print
    print "Event", i
    print
    print "Trajectories:", len(alltraj)
    #print "Merged trj:", nummerged
    #print "Pruned trj:", numpruned
    
    print "Starting voxels:", oldlen
    print "Used voxels:", usedlen
    print "Orphan voxels:", len(run.getOrphans())
    print
    
    totalvox += oldlen
    totaltraj += len(alltraj)
    totalorphans += len(run.getOrphans())

    plotEvent(run, i, len(alltraj), usedlen, len(run.getOrphans()), True, str(i))
    
print    
print "RUN INFORMATION"
print "Total voxels:", totalvox
print "Total trajectories:", totaltraj
print "Total orphans:", totalorphans