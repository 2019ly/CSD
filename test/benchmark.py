import os
import random
import numpy as np

from shapely.geometry import Point

from RkNN import VR_BiRkNN, CSD_MonoRkNN, SLICE_BiRkNN, CSD_BiRkNN, VR_MonoRkNN, SLICE_MonoRkNN
from common.Rtree import RtreeIndex
from common.VoKDtree import VoKDtreeIndex
from common.VoRtree import VoRtreeIndex


def generate_points(n, distribution='Uniform', bounds=None):
    if distribution == 'Uniform':
        if bounds is None:
            bounds = (0, 0, 10000, 10000)
        p_set = set()
        data = []
        while len(p_set) < n:
            x = np.random.uniform(bounds[0], bounds[2])
            y = np.random.uniform(bounds[1], bounds[3])
            if (x, y) not in p_set:
                p_set.add((x, y))
                data.append((len(p_set) - 1, Point(x, y)))
        return data

    if distribution == 'Normal':
        if bounds is None:
            bounds = (0, 0, 10000, 10000)
        p_set = set()
        x_mu = (bounds[0] + bounds[2]) / 2
        x_sigma = abs(bounds[0] - bounds[2]) / 10
        y_mu = (bounds[1] + bounds[3]) / 2
        y_sigma = abs(bounds[1] - bounds[3]) / 10
        data = []
        while len(p_set) < n:
            x = np.random.normal(x_mu, x_sigma)
            y = np.random.normal(y_mu, y_sigma)
            if (x, y) not in p_set and bounds[0] < x < bounds[2] and bounds[1] < y < bounds[3]:
                p_set.add((x, y))
                data.append((len(p_set) - 1, Point(x, y)))
        return data

    if distribution == 'Real':
        if bounds is None:
            bounds = (-125.733611, 23.546944, -65.95, 50.361794)
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data/us50000.txt')
        f = open(path)
        point_pool = list()
        visited = set()
        for line in f.readlines():
            line_str = line.strip().split()
            x = float(line_str[1])
            y = float(line_str[0])
            if (x, y) not in visited:
                visited.add((x, y))
                point_pool.append((x, y))
        points = random.sample(point_pool, n)
        data = [(i, Point(points[i][0], points[i][1])) for i in range(len(points))]
        return data


def generate_index(index_type, n, distribution='Uniform', bounds=None):
    if bounds is None:
        if distribution == 'Real':
            bounds = (-125.733611, 23.546944, -65.95, 50.361794)
        else:
            bounds = (0, 0, 10000, 10000)
    points = generate_points(n, distribution, bounds)
    if index_type == 'Rtree':
        return RtreeIndex(bounds[0], bounds[1], bounds[2], bounds[3], points)
    elif index_type == 'VoRtree':
        return VoRtreeIndex(bounds[0], bounds[1], bounds[2], bounds[3], points)
    elif index_type == 'VoKDtree':
        return VoKDtreeIndex(bounds[0], bounds[1], bounds[2], bounds[3], points)
    return None


def test_mono_rknn(algorithm, index, k, times):
    total_pruning_cost = 0
    total_verification_cost = 0
    total_pruning_io = 0
    total_verification_io = 0
    total_candidate_num = 0
    total_verified_candidate_num = 0
    width = 152
    print '-' * width
    print '| Instance | Pruning time (ms) | Pruning IO | Verification time (ms) | Verification IO | Total time (' \
          'ms) | Total IO | #candidate | #verified candidate |'
    print '-' * width
    for i in range(times):
        q_id = random.randint(0, index.size - 1)
        result, pruning_cost, verification_cost, pruning_io, verification_io, candidate_num, verified_candidate_num = algorithm(
            q_id, k, index, True)
        total_pruning_cost += pruning_cost
        total_verification_cost += verification_cost
        total_pruning_io += pruning_io
        total_verification_io += verification_io
        total_candidate_num += candidate_num
        total_verified_candidate_num += verified_candidate_num
        print '| {:^8} | {:>17.2f} | {:>10} | {:>22.2f} | {:>15} | {:>15.2f} | {:>8} | {:>10} | {:>19} |'.format(
            i + 1,
            pruning_cost * 1000, pruning_io,
            verification_cost * 1000, verification_io,
            pruning_cost * 1000 + verification_cost * 1000, pruning_io + verification_io, candidate_num,
            verified_candidate_num)
    print '-' * width
    print '|  Average | {:>17.2f} | {:>10.2f} | {:>22.2f} | {:>15.2f} | {:>15.2f} | {:>8.2f} | {:>10.2f} | {:>19.2f} |'.format(
        total_pruning_cost / times * 1000, total_pruning_io / float(times),
        total_verification_cost / times * 1000, total_verification_io / float(times),
        total_pruning_cost / times * 1000 + total_verification_cost / times * 1000,
        total_pruning_io / float(times) + total_verification_io / float(times), total_candidate_num / float(times),
        total_verified_candidate_num / float(times))
    print '-' * width


def test_bi_rknn(algorithm, facility_index, user_index, k, times):
    total_pruning_cost = 0
    total_verification_cost = 0
    total_pruning_io = 0
    total_verification_io = 0
    total_candidate_num = 0
    total_verified_candidate_num = 0
    width = 152
    print '-' * width
    print '| Instance | Pruning time (ms) | Pruning IO | Verification time (ms) | Verification IO | Total time (' \
          'ms) | Total IO | #candidate | #verified candidate |'
    print '-' * width
    for i in range(times):
        q_id = random.randint(0, facility_index.size - 1)
        result, pruning_cost, verification_cost, pruning_io, verification_io, candidate_num, verified_candidate_num = algorithm(
            q_id, k, facility_index, user_index, True)
        total_pruning_cost += pruning_cost
        total_verification_cost += verification_cost
        total_pruning_io += pruning_io
        total_verification_io += verification_io
        total_candidate_num += candidate_num
        total_verified_candidate_num += verified_candidate_num
        print '| {:^8} | {:>17.2f} | {:>10} | {:>22.2f} | {:>15} | {:>15.2f} | {:>8} | {:>10} | {:>19} |'.format(
            i + 1,
            pruning_cost * 1000, pruning_io,
            verification_cost * 1000, verification_io,
            pruning_cost * 1000 + verification_cost * 1000, pruning_io + verification_io, candidate_num,
            verified_candidate_num)
    print '-' * width
    print '|  Average | {:>17.2f} | {:>10.2f} | {:>22.2f} | {:>15.2f} | {:>15.2f} | {:>8.2f} | {:>10.2f} | {:>19.2f} |'.format(
        total_pruning_cost / times * 1000, total_pruning_io / float(times),
        total_verification_cost / times * 1000, total_verification_io / float(times),
        total_pruning_cost / times * 1000 + total_verification_cost / times * 1000,
        total_pruning_io / float(times) + total_verification_io / float(times), total_candidate_num / float(times),
        total_verified_candidate_num / float(times))
    print '-' * width


def Effect_of_k_on_MonoRkNN_in_real_distribution(times):
    print('Effect of k on BiRkNN in real distribution')
    data_size = 49601
    distribution = 'Real'
    vokdtree_index = generate_index('VoKDtree', data_size, distribution)
    vortree_index = generate_index('VoRtree', data_size, distribution)
    k_list = [1, 10, 100, 1000]
    for k in k_list:
        print 'k =', k
        print 'CSD-RkNN'
        test_mono_rknn(CSD_MonoRkNN, vokdtree_index, k, times)
        print 'SLICE'
        test_mono_rknn(SLICE_MonoRkNN, vortree_index, k, times)
        print 'VR-RkNN'
        test_mono_rknn(VR_MonoRkNN, vortree_index, k, times)


def Effect_of_k_on_MonoRkNN_in_uniform_distribution(times):
    print('Effect of k on BiRkNN in uniform distribution')
    data_size = 100000
    distribution = 'Uniform'
    vokdtree_index = generate_index('VoKDtree', data_size, distribution)
    vortree_index = generate_index('VoRtree', data_size, distribution)
    k_list = [1, 10, 100, 1000]
    for k in k_list:
        print 'k =', k
        print 'CSD-RkNN'
        test_mono_rknn(CSD_MonoRkNN, vokdtree_index, k, times)
        print 'SLICE'
        test_mono_rknn(SLICE_MonoRkNN, vortree_index, k, times)
        print 'VR-RkNN'
        test_mono_rknn(VR_MonoRkNN, vortree_index, k, times)


def Effect_of_k_on_BiRkNN_in_real_distribution(times):
    print('Effect of k on BiRkNN in real distribution')
    data_size = 49601
    distribution = 'Real'
    vokdtree_facility_index = generate_index('VoKDtree', data_size, distribution)
    vokdtree_user_index = generate_index('VoKDtree', data_size, distribution)
    vortree_facility_index = generate_index('VoRtree', data_size, distribution)
    vortree_user_index = generate_index('VoRtree', data_size, distribution)
    k_list = [1, 10, 100, 1000]
    for k in k_list:
        print 'k =', k
        print 'CSD-RkNN'
        test_bi_rknn(CSD_BiRkNN, vokdtree_facility_index, vokdtree_user_index, k, times)
        print 'SLICE'
        test_bi_rknn(SLICE_BiRkNN, vortree_facility_index, vortree_user_index, k, times)
        print 'VR-RkNN'
        test_bi_rknn(VR_BiRkNN, vortree_facility_index, vortree_user_index, k, times)


def Effect_of_k_on_BiRkNN_in_uniform_distribution(times):
    data_size = 100000
    distribution = 'Uniform'
    vokdtree_facility_index = generate_index('VoKDtree', data_size, distribution)
    vokdtree_user_index = generate_index('VoKDtree', data_size, distribution)
    vortree_facility_index = generate_index('VoRtree', data_size, distribution)
    vortree_user_index = generate_index('VoRtree', data_size, distribution)
    k_list = [1, 10, 100, 1000]
    for k in k_list:
        print 'k = {}'.format(k)
        print 'CSD-RkNN'
        test_bi_rknn(CSD_BiRkNN, vokdtree_facility_index, vokdtree_user_index, k, times)
        print 'SLICE'
        test_bi_rknn(SLICE_BiRkNN, vortree_facility_index, vortree_user_index, k, times)
        print 'VR-RkNN'
        test_bi_rknn(VR_BiRkNN, vortree_facility_index, vortree_user_index, k, times)


def Effect_of_data_size_on_MonoRkNN(times):
    print('Effect of data size on MonoRkNN')
    k = 100
    data_size_list = [1000, 10000, 100000, 1000000]
    for data_size in data_size_list:
        print 'Data size =', data_size
        vokdtree_index = generate_index('VoKDtree', data_size, 'Uniform')
        vortree_index = generate_index('VoRtree', data_size, 'Uniform')
        print 'CSD-RkNN'
        test_mono_rknn(CSD_MonoRkNN, vokdtree_index, k, times)
        print 'SLICE'
        test_mono_rknn(SLICE_MonoRkNN, vortree_index, k, times)
        print 'VR-RkNN'
        test_mono_rknn(VR_MonoRkNN, vortree_index, k, times)


def Effect_of_data_size_on_BiRkNN(times):
    print('Effect of data size on BiRkNN')
    k = 100
    data_size_list = [1000, 10000, 100000, 1000000]
    for data_size in data_size_list:
        print 'Data size =', data_size
        vokdtree_facility_index = generate_index('VoKDtree', data_size, 'Uniform')
        vokdtree_user_index = generate_index('VoKDtree', data_size, 'Uniform')
        vortree_facility_index = generate_index('VoRtree', data_size, 'Uniform')
        vortree_user_index = generate_index('VoRtree', data_size, 'Uniform')
        print 'CSD-RkNN'
        test_bi_rknn(CSD_BiRkNN, vokdtree_facility_index, vokdtree_user_index, k, times)
        print 'SLICE'
        test_bi_rknn(SLICE_BiRkNN, vortree_facility_index, vortree_user_index, k, times)
        print 'VR-RkNN'
        test_bi_rknn(VR_BiRkNN, vortree_facility_index, vortree_user_index, k, times)


def Effect_of_user_num_relative_to_facility_num(times):
    print('Effect of |U|/|F|')
    k = 100
    facility_num = 100000
    user_per_facility = [0.5, 1, 2, 4]
    vokdtree_facility_index = generate_index('VoKDtree', facility_num, 'Uniform')
    vortree_facility_index = generate_index('VoRtree', facility_num, 'Uniform')
    for i in user_per_facility:
        user_num = i * facility_num
        print '|U|/|F| = {:.0%}'.format(i)
        vokdtree_user_index = generate_index('VoKDtree', user_num, 'Uniform')
        vortree_user_index = generate_index('VoRtree', user_num, 'Uniform')
        print 'CSD-RkNN'
        test_bi_rknn(CSD_BiRkNN, vokdtree_facility_index, vokdtree_user_index, k, times)
        print 'SLICE'
        test_bi_rknn(SLICE_BiRkNN, vortree_facility_index, vortree_user_index, k, times)
        print 'VR-RkNN'
        test_bi_rknn(VR_BiRkNN, vortree_facility_index, vortree_user_index, k, times)


def Effect_of_distribution(times):
    print('Effect of distribution')
    k = 100
    distributions = [('Uniform', 'Uniform'), ('Uniform', 'Normal'), ('Uniform', 'Real'), ('Normal', 'Uniform'),
                     ('Normal', 'Normal'), ('Normal', 'Real'), ('Real', 'Uniform'), ('Real', 'Normal'),
                     ('Real', 'Real')]
    data_size = 100000
    for d in distributions:
        print 'user distribution: {}, facility distribution: {}'.format(d[0], d[1])
        vokdtree_facility_index = generate_index('VoKDtree', data_size, d[1])
        vokdtree_user_index = generate_index('VoKDtree', data_size, d[0])
        vortree_facility_index = generate_index('VoRtree', data_size, d[1])
        vortree_user_index = generate_index('VoRtree', data_size, d[0])
        print 'CSD-RkNN'
        test_bi_rknn(CSD_BiRkNN, vokdtree_facility_index, vokdtree_user_index, k, times)
        print 'SLICE'
        test_bi_rknn(SLICE_BiRkNN, vortree_facility_index, vortree_user_index, k, times)
        print 'VR-RkNN'
        test_bi_rknn(VR_BiRkNN, vortree_facility_index, vortree_user_index, k, times)
