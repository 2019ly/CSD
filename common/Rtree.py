# -*- coding:utf-8 -*-

import random
from heapq import heappop, heappush

from shapely.geometry import box, Point, Polygon
from shapely.ops import unary_union
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import common


def bounding_box(geoms):
    return unary_union(geoms).envelope


class RtreeIndex(object):
    def __init__(self, minx, miny, maxx, maxy, data):
        self.count = 0
        self.leaf_count = 0
        self.MAX_CHILDREN_NUM = 10
        self.MIN_CHILDREN_NUM = int(self.MAX_CHILDREN_NUM / 2)
        self.CLUSTER_NUM = 2
        self.root = Node.create(self, None, Polygon(), [])
        self.space = box(minx, miny, maxx, maxy)
        self.geometries = dict()
        self.points = self.geometries
        self.size = len(data)
        for e in data:
            self.insert(e[0], e[1])

    def insert(self, o, geometry):
        root = self.root.insert(o, geometry)
        if len(root) > 1:
            self.root = Node.create_with_children(self, root)
        self.geometries[o] = geometry

    def get_all(self, node=None):
        if node is None:
            node = self.root
        yield node.obj, node.geom
        if not node.is_data_node:
            for c in node.children:
                for n in self.get_all(c):
                    yield n

    def range(self, r):
        for x in self.root.search(lambda x: r.intersects(x.geom), lambda x: r.intersects(x.geom)):
            yield x.obj, x.geom

    def nearest(self, q, k=1):
        min_heap = common.MinHeap()
        knn = common.MaxHeap()
        min_heap.push((0, self.root))
        while len(min_heap) > 0:
            node_min_dist, node = min_heap.pop()
            if len(knn) >= k and node_min_dist > knn.first()[0]:
                break
            if node.is_leaf_node:
                for c in node.children:
                    c_min_dist = c.geom.distance(q)
                    if len(knn) < k or c_min_dist < knn.first()[0]:
                        knn.push((c_min_dist, c))
                        if len(knn) > k:
                            knn.pop()
            else:
                for c in node.children:
                    c_min_dist = c.geom.distance(q)
                    if len(knn) < k or c_min_dist < knn.first()[0]:
                        min_heap.push((c_min_dist, c))
        return [(e[1].obj, e[1].geom, e[0]) for e in knn.values]


class Node(object):
    @classmethod
    def create(cls, tree, obj, geom, children):
        return Node(tree, obj, geom, children)

    @classmethod
    def create_with_children(cls, tree, children):
        geom = bounding_box([c.geom for c in children])
        node = Node.create(tree, None, geom, children)
        assert (not node.is_data_node)
        return node

    @classmethod
    def create_data_node(cls, tree, obj, geom):
        node = Node.create(tree, obj, geom, None)
        assert node.is_data_node
        return node

    def __init__(self, tree, obj, geom, children):
        self.tree = tree
        self.obj = obj
        self.geom = geom
        self.children = children

    def search(self, intermediate_node_predicate, data_node_predicate):
        if self.is_data_node:
            if data_node_predicate(self):
                yield self
        else:
            if intermediate_node_predicate(self):
                for c in self.children:
                    for cr in c.search(intermediate_node_predicate, data_node_predicate):
                        yield cr

    @property
    def is_data_node(self):
        if self.children is not None:
            return False
        if self.geom is not None and self.obj is not None:
            return True
        return False

    @property
    def is_leaf_node(self):
        if self.is_data_node:
            return False
        if len(self.children) == 0 or self.children[0].is_data_node:
            return True
        return False

    @property
    def is_intermediate_node(self):
        if (not self.is_data_node) and (not self.is_leaf_node):
            return True

    def insert(self, obj, geom):
        self.geom = bounding_box([self.geom, geom])
        if self.is_leaf_node:
            self.children.append(Node.create_data_node(self.tree, obj, geom))
            return self.adjust()
        else:
            inserting_child = self.find_inserting_child(geom)
            new_children = inserting_child.insert(obj, geom)
            if len(new_children) > 1:
                self.children.remove(inserting_child)
                self.children += new_children
            return self.adjust()

    def find_inserting_child(self, geom):
        min_enlargement = float('inf')
        min_area = float('inf')
        inserting_child = None
        for e in self.children:
            area = bounding_box([e.geom, geom]).area
            enlargement = area - e.geom.area
            if enlargement < min_enlargement:
                inserting_child = e
                min_enlargement = enlargement
                min_area = area
            elif enlargement == min_enlargement:
                if area < min_area:
                    inserting_child = e
                    min_area = area
        return inserting_child

    def adjust(self):
        if len(self.children) <= self.tree.MAX_CHILDREN_NUM:
            return [self]
        return self.split()

    def quadratic_split(self):
        seeds, remain_entries = self.pick_seeds()
        groups = [[seeds[0]], [seeds[1]]]
        group_bounding_boxes = [seeds[0].geom, seeds[1].geom]
        while len(remain_entries) > 0:
            if len(groups[0]) > self.tree.MIN_CHILDREN_NUM:
                groups[1] += remain_entries
                break
            if len(groups[1]) > self.tree.MIN_CHILDREN_NUM:
                groups[0] += remain_entries
                break
            e, bounding_boxes = self.pick_next(remain_entries, group_bounding_boxes)
            areas = [bounding_boxes[0].area, bounding_boxes[1].area]
            area_differences = [areas[0] - group_bounding_boxes[0].area, areas[1] - group_bounding_boxes[1].area]
            if area_differences[0] < area_differences[1]:
                target_group_id = 0
            elif area_differences[0] > area_differences[1]:
                target_group_id = 1
            else:
                if areas[0] < areas[1]:
                    target_group_id = 0
                elif areas[0] > areas[1]:
                    target_group_id = 1
                else:
                    if len(groups[0]) <= len(groups[1]):
                        target_group_id = 0
                    else:
                        target_group_id = 1
            groups[target_group_id].append(e)
            group_bounding_boxes[target_group_id] = bounding_boxes[target_group_id]
        return [Node.create_with_children(self.tree, group) for group in groups]

    def pick_seeds(self):
        seeds = []
        area_difference = -float('inf')
        for i in range(0, len(self.children) - 1):
            for j in range(i + 1, len(self.children)):
                area = bounding_box([self.children[i].geom, self.children[j].geom]).area - self.children[i].geom.area - \
                       self.children[j].geom.area
                if area > area_difference:
                    seeds = (self.children[i], self.children[j])
                    area_difference = area
        remain_entries = []
        for e in self.children:
            if e not in seeds:
                remain_entries.append(e)
        return seeds, remain_entries

    @staticmethod
    def pick_next(remain_entries, group_bounding_boxes):
        difference = -1
        for i in range(len(remain_entries)):
            bbox1 = bounding_box([group_bounding_boxes[0], remain_entries[i].geom])
            bbox2 = bounding_box([group_bounding_boxes[1], remain_entries[i].geom])
            d1 = bbox1.area - group_bounding_boxes[0].area
            d2 = bbox2.area - group_bounding_boxes[1].area
            if abs(d1 - d2) > difference:
                difference = abs(d1 - d2)
                next_entry_id = i
                next_bounding_boxes = [bbox1, bbox2]
        next_entry = remain_entries.pop(next_entry_id)
        return next_entry, next_bounding_boxes

    split = quadratic_split


def k_means_cluster(k, nodes):
    if len(nodes) <= k:
        return [[node] for node in nodes]
    centers = [node.geom.centroid.coords[0] for node in nodes]
    clustering = KMeans(n_clusters=k)
    clustering.fit(centers)
    labels = clustering.labels_
    return [[nodes[j] for j in range(len(nodes)) if labels[j] == i] for i in range(k)]


def plot_geometry(ax, geometry, color='gray'):
    if isinstance(geometry, Point):
        ax.plot(geometry.x, geometry.y, '.', color='red')
    else:
        coordinates = list(geometry.exterior.coords)
        ax.plot([c[0] for c in coordinates], [c[1] for c in coordinates], '-', color=color)
