import copy
import csv
import pickle
import random

import numpy as np
import time

data_mat_const = pickle.load(open('Hospital_dataset.pickle', 'rb')).astype(np.int64)
ground_truth = np.asarray(list(csv.reader(open('Complications - Hospital - Encoded.csv'))), dtype=int)

numS, numE, numA = data_mat_const.shape[0], data_mat_const.shape[1], data_mat_const.shape[2]

max_it = 20

def gen_dictionary(data_list):
    rtn_list = [{} for x in range(len(data_list[0]))]
    reverse_list = [{} for x in range(len(data_list[0]))]
    data_mat_ = np.asarray(data_list)

    for i in range(data_mat_.shape[1]):
        seq = 0
        col_list = data_mat_[:, i].tolist()
        for j in range(data_mat_.shape[0]):
            if col_list[j] not in rtn_list[i]:
                rtn_list[i][col_list[j]] = seq
                reverse_list[i][seq] = col_list[j]
                seq += 1

    return rtn_list, reverse_list


def encoding_dataset(data_list, data_dict):
    for i in range(len(data_list)):
        for j in range(len(data_list[1])):
            data_list[i][j] = data_dict[j][data_list[i][j]]
    return data_list


def encoding_truth(data_list, data_dict):
    for i in range(len(data_list)):
        for j in range(len(data_list[1])):
            try:
                data_list[i][j] = data_dict[j + 1][data_list[i][j]]
            except KeyError:
                data_list[i][j] = len(data_dict[j].keys()) + 1
                # print(j)
    return data_list


def load_data(data_file_name, truth_file_name):

    fp1 = open(data_file_name, errors='ignore', mode='r')
    fp2 = open(truth_file_name, errors='ignore', mode='r')

    restaurant_list = list(csv.reader(fp1))
    del (restaurant_list[0])

    truth_list = list(csv.reader(fp2))
    del (truth_list[0])

    # Generate the dictionary for each attribute
    restaurant_dict, reverse_dict = gen_dictionary(restaurant_list)

    '''Replace claims with codes'''
    restaurant_list = np.asarray(encoding_dataset(restaurant_list, restaurant_dict), dtype=int)
    truth_list = np.asarray(encoding_truth(truth_list, restaurant_dict), dtype=int)

    fp1.close()
    fp2.close()

    numS_, numE_, numA_ = len(restaurant_dict[0]), len(restaurant_dict[-1]), truth_list.shape[1] - 1

    restaurant_mat = np.ones(shape=(numS_, numE_, numA_), dtype=int) * -1
    truth_mat = np.ones(shape=(numE_, numA_), dtype=int) * -1
    claim_mat_ = np.ones(shape=(numS_, numE_), dtype=int) * -1

    for i in range(restaurant_list.shape[0]):
        s, o = restaurant_list[i][0], restaurant_list[i][-1]
        claim_mat_[s, o] = 1
        restaurant_mat[s, o, :] = restaurant_list[i][1:-1]

    for i in range(truth_list.shape[0]):
        o = truth_list[i][-1]
        truth_mat[o, :] = truth_list[i][0:-1]

    return restaurant_mat, truth_mat, claim_mat_, \
           len(restaurant_dict[0]), len(restaurant_dict[-1]), truth_list.shape[1] - 1, \
           reverse_dict

def evaluate(data_mat_con, truth_val, n_answered):
    errors = 0
    for e in range(numE):
        if np.max(data_mat_con[:, e]) > 0:
            for a in range(8, 10):
                errors = errors + np.count_nonzero(truth_val[e, a] - ground_truth[e, a])

    return errors / (n_answered * 2)


if __name__ == '__main__':
    # Parameters
    start = time.clock()
    truth = np.zeros(shape=(numE, numA), dtype=np.int64)
    num_Claims = np.zeros(shape=(numS,), dtype=np.int64)
    claim_confident = [[1.0 for a in range(numA)] for e in range(numE)]
    cleaned_set = []
    num_answered = 0  # Number of claimed entities
    num_errors = 0
    total_claims = 0

    knowledge_pairs = [(8, 9)]
    # knowledge_pairs = [(0, 1)]

    # Voting Initialization
    for e in range(numE):
        if np.max(data_mat_const[:, e]) > 0:
            num_answered += 1

    for s in range(numS):
        for e in range(numE):
            if not np.all(data_mat_const[s][e] == -1):
                num_Claims[s] += 1

    # Count num_Claims
    for s in range(numS):
        for e in range(numE):
            if not np.all(data_mat_const[s][e] == -1):
                for a in range(8, 10):
                    if data_mat_const[s][e][a] != ground_truth[e][a]:
                        num_errors += 1
                total_claims += 1

    # Evaluate the result
    #print('Error rate: {0}'.format(evaluate(data_mat_const, truth, num_answered)))
    print('Data Error rate: {0}'.format(num_errors / (total_claims * 2)))

    num_changed = 0
    num_change2_true = 0
    whole_set = []
    for a in range(30000, 60000):
        whole_set.append(a)

    for pair in knowledge_pairs:
        print(pair)
        att_claim_list = data_mat_const[:, :, [pair[0], pair[1]]].tolist()
        att_claim_list_raw = data_mat_const[:, :, [pair[0], pair[1]]].tolist()
        unique_0 = np.unique(data_mat_const[:, :, pair[0]], return_index=False)
        unique_0 = unique_0.tolist()
        determine_set = []
        remained_set = copy.copy(unique_0)
        fact_list = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10}
        #fact_list = {}
        samples = np.random.uniform(0, 1, numE * numA)
        i = 0
        for e in range(numE):
            for s in range(numS):
                if np.max(data_mat_const[s, e]) > 0:
                    # print(s, e, pair[1])
                    # print(fact_list)
                    if samples[i] < 1:  # Change dependence attribute
                        if att_claim_list[s][e][0] in fact_list:
                            # print(truth[s][e][pair[0]])
                            # print(truth[s][e][pair[1]])

                            # print(truth[s][e][0])

                            # print(fact_list[truth[s][e][pair[0]]])
                            if fact_list[att_claim_list[s][e][0]] != att_claim_list[s][e][1]:
                                # num_changed += 1
                                # if fact_list[att_claim_list[s][e][0]] == ground_truth[e][pair[1]]:
                                #     num_change2_true += 1
                                att_claim_list[s][e][1] = fact_list[att_claim_list[s][e][0]]
                                # num_changed += 1
                                # if att_claim_list[s][e][1] == ground_truth[e][pair[1]]:
                                #     num_change2_true += 1
                        else:
                            fact_list[att_claim_list[s][e][0]] = att_claim_list[s][e][1]
                            # determine_set.append(att_claim_list[s][e][0])
                            # remained_set.remove(att_claim_list[s][e][0])
                    else:  # Change determine attribute
                        if att_claim_list[s][e][0] in fact_list:
                            if fact_list[att_claim_list[s][e][0]] != att_claim_list[s][e][1]:
                                att_claim_list[s][e][0] = random.choice(whole_set)
                                # num_changed += 1
                                # if att_claim_list[s][e][0] == ground_truth[e][pair[0]]:
                                #     num_change2_true += 1
                        else:
                            fact_list[att_claim_list[s][e][0]] = att_claim_list[s][e][1]

                            # fact_list[truth[e][pair[0]]] = truth[e][pair[1]]
            i += 1
        print(fact_list)

    for e in range(numE):
        if np.max(data_mat_const[:, e]) > 0:
            for a in range(8, 10):
                claim_list = []
                claim_list_raw = []
                for s in range(numS):
                    claim_list.append(att_claim_list[s][e][a-8])
                    claim_list_raw.append(att_claim_list_raw[s][e][a - 8])
                claim_list = np.asarray([x for x in claim_list if x != -1])
                truth[e][a] = np.argmax(np.bincount(claim_list))
                for ii in range(len(claim_list_raw)):
                    if claim_list_raw[ii] != -1 and claim_list_raw[ii] != truth[e][a]:
                        num_changed += 1
                        if truth[e][a] == ground_truth[e, a]:
                            num_change2_true += 1

    end = time.clock()
    print(end - start)
    print('Precision: {0}'.format(num_change2_true / num_changed))
    print('Recall: {0}'.format(num_change2_true / num_errors))

    # Evaluate the result
    print('Error rate: {0}'.format(evaluate(data_mat_const, truth, num_answered)))
