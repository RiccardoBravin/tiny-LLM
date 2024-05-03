import torch
import matplotlib.pyplot as plt
import math

leng = 128
#make a list of numbers from 0 to 1 with steps of 1/256
x = torch.arange(0, 1, 1/leng) * torch.pi 
x1 = x

#a=10
#b=1
r = 1
c = 20
#spiral on sphere
# points = torch.stack([r*torch.sin(x)*torch.cos(x*c), r*torch.sin(x)*torch.sin(x*c), r*torch.cos(x)], dim=1)
#spiral on sphere with linear z axis
# points = torch.stack([r*torch.sin(x)*torch.cos(x*c), r*torch.sin(x)*torch.sin(x*c), r*x - torch.pi/2], dim=1)
#spiral on cilinder
#points = torch.stack([r*torch.cos(x*c), r*torch.sin(x*c), r*x - torch.pi/2], dim=1) #satisfies property 1 and 2 in 3d
#guessing good function to try
#points = torch.stack([r*x1*torch.cos(x1*c), r*x1*torch.sin(x1*c), r*torch.cos(x1)], dim=1)

d = 0.04
points = torch.tensor([[0,0,1]])
delta = 0.0001
i = 0
while len(points) < leng:
    pos = delta*i
    p = torch.tensor([r*math.sin(pos)*math.cos(pos*c), r*math.sin(pos)*math.sin(pos*c), r*math.cos(pos)])
    if torch.norm(p - points[-1]) > d:
        points = torch.cat((points, p.unsqueeze(0)), dim=0)
    i += 1

print(points)

#plot points in a 3d graph
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:,0], points[:,1], points[:,2])
plt.show()

#on a new  2d graph show the distance between f(0) and f(x)
distances = torch.norm(points - points[0], dim=1)
plt.title('Distance from first point')
plt.plot(x, distances)
plt.show()

#on a new 2d graph show the distance between one point and the next one
diff = 20
distances = torch.norm(points[diff:] - points[:-diff], dim=1)
#round the distances to 3 decimal places
distances = torch.round(distances*100000)/100000
plt.title('Distance between points')
plt.plot(x[:-diff], distances)
plt.show()

# distances = torch.norm(points, dim=1)
# #round the distances to 3 decimal places
# distances = torch.round(distances*100000)/100000
# plt.title('Distance from origin')
# plt.plot(x, distances)
# plt.show()


# import math


# def fibonacci_sphere(samples=1000):

#     points = []
#     phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians

#     for i in range(samples):
#         y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
#         radius = math.sqrt(1 - y * y)  # radius at y

#         theta = phi * i  # golden angle increment

#         x = math.cos(theta) * radius
#         z = math.sin(theta) * radius

#         points.append((x, y, z))

#     return points



# points = fibonacci_sphere(128)
# points = torch.tensor(points)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# #connect points with line
# ax.plot(points[:,0], points[:,1], points[:,2])
# #ax.scatter(points[:,0], points[:,1], points[:,2])
# plt.show()

# distances = torch.norm(points, dim=1)
# distances = torch.round(distances*100000)/100000
# plt.plot(distances)
# plt.title('Distance from origin')
# plt.show()

# distances = torch.norm(points - points[0], dim=1)
# distances = torch.round(distances*100000)/100000
# plt.title('Distance from first point')
# plt.plot(distances)
# plt.show()

# diff = 2
# distances = torch.norm(points[diff:] - points[:-diff], dim=1)
# distances = torch.round(distances*100000)/100000
# plt.title('Distance between points')
# plt.plot(distances)
# plt.show()

























""" 
import math
def sphericalcoordinate(x,y):
    return [math.cos( x ) * math.cos(y), math.sin(x) * math.cos(y), math.sin(y)]

def NX( n , x ) :
    pts =[]
    start = (-1 + 1/(n - 1))
    increment = (2 - 2 / (n - 1 )) / (n - 1)
    for j in range ( 0 , n ) :
        s = start + j * increment
        pts.append (
                sphericalcoordinate( s * x , math.pi / 2 * math.copysign( 1 , s ) * ( 1 - math.sqrt( 1 - abs(s)))) 
            )
    return pts

def generatepoints(n):
    return NX(n , 0.1 + 1.2 * n) 

"""