import torch

def fillFalse(check_tensor_origin, first_row, column):

    check_tensor = check_tensor_origin.detach().clone()

    queue = [first_row]

    while len(queue) != 0:
        row = queue.pop(0)
        check_vector = check_tensor[row]
        if sum(check_vector) == 0:
            pass
        else:
            for i in range(check_vector.shape[0]):
                if check_vector[i] == True:

                    if i != row:
                        check_tensor[i, row] = False
                        check_tensor[row, i] = False
                        queue.append(i)
                    
                    if i == column:
                        check_tensor_origin[row, i] = False
                        check_tensor_origin[i, row] = False

    pass


# check_matrix = torch.tensor([   [True, True, True, False, False],
#                                 [True, True, False, True, False],
#                                 [True, False, True, True, False],
#                                 [False, True, True, True, True],
#                                 [False, False, False, True, True]])

# fillFalse(check_matrix, 0, 3)


for i in range(10):
    print(i)
    i-=1
