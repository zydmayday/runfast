import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


if __name__ == '__main__':
    PLAYER_KEY = ['nn_train1', 'nn_train2', 'nn_train3']
    # PLAYER_KEY = ['dn_train1', 'dn_train2', 'dn_train3']
    # PLAYER_KEY = ['dn_memory_train1', 'dn_memory_train2', 'dn_memory_train3']
    # test_file = 'test_winners_dn_with_momery'
    test_file = 'test_winners_nn'
    test_data = pickle.load(open(test_file))
    x_axis_list = test_data.keys()
    x_axis_list.sort()
    y1_list = []
    y2_list = []
    y3_list = []
    for k in x_axis_list:
        item = test_data[k]
        y1_list.append(item[PLAYER_KEY[0]]['win'])
        y2_list.append(item[PLAYER_KEY[1]]['win'])
        y3_list.append(item[PLAYER_KEY[2]]['win'])

    # red_patch = mpatches.Patch(color='red', label='trainer1')
    # blue_patch = mpatches.Patch(color='blue', label='trainer2')
    # green_patch = mpatches.Patch(color='green', label='trainer3')
    # plt.legend(handles=[red_patch])
    # plt.plot(x_axis_list, y1_list, 'r-', x_axis_list, y2_list, 'b-', x_axis_list, y3_list, 'g-')
    plt.plot(x_axis_list, y1_list, 'r-', label='trainer1')
    plt.plot(x_axis_list, y2_list, 'b-', label='trainer2')
    plt.plot(x_axis_list, y3_list, 'g-', label='trainer3')
    legend = plt.legend(loc='upper center', shadow=True)
    # plt.show()
    plt.savefig('nn_test_1000_50_10000_norewardinside.png')