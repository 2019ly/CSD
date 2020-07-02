from shapely.geometry import Point

import common
from common.Rtree import RtreeIndex
from common.VoronoiDiagram import VoronoiDiagram


class VoRtreeIndex(RtreeIndex):
    def __init__(self, minx, miny, maxx, maxy, data):
        super(VoRtreeIndex, self).__init__(minx, miny, maxx, maxy, data)
        self.voronoi_diagram = VoronoiDiagram(data, (minx, miny), (maxx, maxy))
        self.points = self.geometries

    def NN(self, q):
        IO = 0
        h = common.MinHeap()
        h.push((0, self.root))
        IO += 1
        best_dist = float('inf')
        nn = None
        while len(h) > 0:
            e_dist, e = h.pop()
            if e.is_leaf_node:
                for child in e.children:
                    IO += 1
                    c_dist = child.geom.distance(q)
                    if c_dist < best_dist:
                        nn = child
                        best_dist = c_dist
                if self.voronoi_diagram.cell(nn.obj).intersects(q):
                    return (nn.obj, nn.geom, best_dist), IO
            else:
                for child in e.children:
                    IO += 1
                    c_dist = child.geom.distance(q)
                    h.push((c_dist, child))
        return (nn.obj, nn.geom, best_dist), IO

    def kNN(self, q, k):
        IO = 0
        knn = list()
        h = common.MinHeap()
        (nn_o, nn_p, nn_dist), nn_io = self.NN(q)
        # nn_o, nn_p, nn_dist = e
        IO += nn_io
        vd = self.voronoi_diagram
        points = self.points
        h.push((nn_dist, nn_o))
        visited = {nn_o}
        count = 0
        while count < k and len(h) > 0:
            dist, o = h.pop()
            p = points[o]
            knn.append((o, p, dist))
            for neighbor in vd.neighbors(o):
                if neighbor not in visited:
                    IO += 1
                    visited.add(neighbor)
                    h.push((points[neighbor].distance(q), neighbor))
            count += 1
        return knn, IO

    def nearest(self, q, k=1):
        if k == 1:
            return self.NN(q)
        else:
            return self.kNN(q, k)
