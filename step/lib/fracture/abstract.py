import numpy
from shapely.geometry import LineString
from sklearn.neighbors import KDTree


def svd(points):
    pts = points.copy()
    # Algorithm note: see docs/METHODOLOGY.md.
    c = numpy.mean(pts, axis=0)
    A = pts - c  # shift the points
    A = A.T  # 3*n
    u, s, vh = numpy.linalg.svd(A, full_matrices=False, compute_uv=True)  # A=u*s*vh
    normal = u[:, -1]
    # Algorithm note: see docs/METHODOLOGY.md.
    normal = normal / numpy.linalg.norm(normal)
    return normal

def estNormals(pts_on_skele, skele_pts, n):  # See docs/METHODOLOGY.md.
    pts = numpy.copy(skele_pts)
    tree = KDTree(pts, leaf_size=2)
    idx = tree.query(pts_on_skele, k=n, return_distance=False, dualtree=False, breadth_first=False)
    normals = []
    for i in range(0, pts_on_skele.shape[0]):
        pts_for_normals = pts[idx[i, :], :]
        normal = svd(pts_for_normals)
        normals.append(normal)
    normals = numpy.array(normals)
    return normals

def getWidths(centers, normals, bpoints, ratio):
    # Algorithm note: see docs/METHODOLOGY.md.
    widths = []
    for i in range(len(centers)):
        # Algorithm note: see docs/METHODOLOGY.md.
        line = LineString([centers[i]+1000*normals[i], centers[i]-1000*normals[i]])  # See docs/METHODOLOGY.md.
        intersect = []
        for j in range(len(bpoints)):
            poly = LineString(bpoints[j]+[bpoints[j][0]])
            intersection = line.intersection(poly)
            if intersection.geom_type == 'MultiPoint':
                for point in intersection.geoms:
                    intersect.append(numpy.array([point.x, point.y]))
        # Algorithm note: see docs/METHODOLOGY.md.
        if len(intersect) == 0:
            continue
        vector = numpy.array(intersect) - centers[i]
        pos = vector[vector.dot(normals[i]) > 0]
        neg = vector[vector.dot(normals[i]) < 0]
        if len(pos)*len(neg) == 0:
            continue
        widths.append((min([numpy.sqrt(j.dot(j)) for j in pos]) + min([numpy.sqrt(j.dot(j)) for j in neg]))*ratio[0])
    return widths


# Algorithm note: see docs/METHODOLOGY.md.
def getAperture(cluster, outline, ratio):
    # Algorithm note: see docs/METHODOLOGY.md.
    iter_num = 10
    cluster = numpy.fliplr(cluster)
    # Algorithm note: see docs/METHODOLOGY.md.
    pts = cluster[range(0, len(cluster), iter_num)]
    # Algorithm note: see docs/METHODOLOGY.md.
    pns = estNormals(pts, cluster, 4)
    # Algorithm note: see docs/METHODOLOGY.md.
    widths = getWidths(pts, pns, outline, ratio)
    # Algorithm note: see docs/METHODOLOGY.md.
    aperture = numpy.mean(widths)
    return aperture


# Algorithm note: see docs/METHODOLOGY.md.
def getTrace(cluster, ratio):
    vector = cluster[:-1] - cluster[1:]
    trace = sum([numpy.sqrt(i.dot(i))*ratio[0] for i in vector])
    return trace


