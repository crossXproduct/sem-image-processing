import torch
import torchvision
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())
import numpy as np
import rdfpy
import matplotlib.pyplot as plt
import cv2
import sys
#sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

#SETUP
filename = str(input('filename: '))
image = cv2.imread(filename)  #Try houses.jpg or neurons.jpg
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10,10))
plt.imshow(image)
plt.axis('off')
plt.show()

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cpu"  # Change to "cpu" to use CPU

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)


#FUNCTIONS
#Mask generator
#There are several tunable parameters in automatic mask generation that control 
# how densely points are sampled and what the thresholds are for removing low 
# quality or duplicate masks. Additionally, generation can be automatically 
# run on crops of the image to get improved performance on smaller objects, 
# and post-processing can remove stray pixels and holes. 
# Here is an example configuration that samples more masks:
#https://github.com/facebookresearch/segment-anything/blob/9e1eb9fdbc4bca4cd0d948b8ae7fe505d9f4ebc7/segment_anything/automatic_mask_generator.py#L35    
#Rerun the following with a few settings, ex. 0.86 & 0.9 for iou_thresh
# and 0.92 and 0.96 for score_thresh
mask_generator_ = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.9,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Requires open-cv to run post-processing
)

# RDF
def pairCorrelationFunction_2D(x, y, S, rMax, dr):
    """Compute the two-dimensional pair correlation function, also known
    as the radial distribution function, for a set of circular particles
    contained in a square region of a plane.  This simple function finds
    reference particles such that a circle of radius rMax drawn around the
    particle will fit entirely within the square, eliminating the need to
    compensate for edge effects.  If no such particles exist, an error is
    returned. Try a smaller rMax...or write some code to handle edge effects! ;)

    Arguments:
        x               an array of x positions of centers of particles
        y               an array of y positions of centers of particles
        S               length of each side of the square region of the plane
        rMax            outer diameter of largest annulus
        dr              increment for increasing radius of annulus

    Returns a tuple: (g, radii, interior_indices)
        g(r)            a numpy array containing the correlation function g(r)
        radii           a numpy array containing the radii of the
                        annuli used to compute g(r)
        reference_indices   indices of reference particles
    """
    from numpy import zeros, sqrt, where, pi, mean, arange, histogram
    # Number of particles in ring/area of ring/number of reference particles/number density
    # area of ring = pi*(r_outer**2 - r_inner**2)

    # Find particles which are close enough to the box center that a circle of radius
    # rMax will not cross any edge of the box
    #bools1 = x > rMax not sure why these conditions exist
    bools2 = x < (S - rMax)
    #bools3 = y > rMax
    bools4 = y < (S - rMax)
    #interior_indices, = where(bools1 * bools2 * bools3 * bools4)
    interior_indices, = where(bools2 * bools4)
    num_interior_particles = len(interior_indices)

    if num_interior_particles < 1:
        raise  RuntimeError ("No particles found for which a circle of radius rMax\
                will lie entirely within a square of side length S.  Decrease rMax\
                or increase the size of the square.")

    edges = arange(0., rMax + 1.1 * dr, dr)
    num_increments = len(edges) - 1
    g = zeros([num_interior_particles, num_increments])
    radii = zeros(num_increments)
    numberDensity = len(x) / S**2

    # Compute pairwise correlation for each interior particle
    for p in range(num_interior_particles):
        index = interior_indices[p]
        d = sqrt((x[index] - x)**2 + (y[index] - y)**2) 
        np.delete(d,np.argwhere(d < 1.00000))
        d[index] = 2 * rMax
        (result, bins) = histogram(d, bins=edges)
        g[p, :] = result/numberDensity

    # Average g(r) for all interior particles and compute radii
    g_average = zeros(num_increments)
    for i in range(num_increments):
        radii[i] = (edges[i] + edges[i+1]) / 2.
        rOuter = edges[i + 1]
        rInner = edges[i]
        g_average[i] = mean(g[:, i]) / (pi * (rOuter**2 - rInner**2))

    return (g_average, radii, edges, interior_indices)

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))


#MAIN
##create masks
print("Generating masks...")
masks = mask_generator_.generate(image)

##save center points to array/file
center = np.zeros((2,len(masks)))
for i in range(0,len(masks)-1):
    center[0][i] = np.average([masks[i]['bbox'][0],masks[i]['bbox'][0]+masks[i]['bbox'][2]])
    center[1][i] = np.average([masks[i]['bbox'][1],masks[i]['bbox'][1]+masks[i]['bbox'][3]])
center_arr = np.array(center)
np.savetxt(filename+'_centers',center_arr)
print('number of masks: ', len(center_arr[0]))
##plot annotated figure
plt.figure(figsize=(10,10))
plt.imshow(image)
show_anns(masks)
plt.scatter(center_arr[:][0],center_arr[:][1],marker='o',color='red')
plt.axis('off')
plt.savefig(filename+'-mask.png')
plt.show()
plt.clf()

##calculate RDF
g_average, radii, edges, interior_indices = pairCorrelationFunction_2D(center_arr[:][0],center_arr[:][1],image.shape[0]*3,image.shape[0],1)
#np.delete(g_average,np.argwhere(d==0.00000000e+00))
print(image.shape[0],image.shape[0]*3/2,image.shape[0]/4)
print(g_average,edges)

##plot RDF
plt.stairs(g_average,edges)
plt.title('RDF of CFoam pores (sample A-1)')
plt.xlabel('distance (pixels)')
plt.ylabel('frequency')
plt.savefig(filename+'-rdf.png')
plt.show()


"""
Mask generation returns a list over masks, where each mask is a dictionary containing various data about the mask. These keys are:

segmentation : the mask
area : the area of the mask in pixels
bbox : the boundary box of the mask in XYWH format
predicted_iou : the model's own prediction for the quality of the mask
point_coords : the sampled input point that generated this mask
stability_score : an additional measure of mask quality
crop_box : the crop of the image used to generate this mask in XYWH format

"""