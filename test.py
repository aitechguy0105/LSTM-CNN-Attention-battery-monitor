def decode(message_file):
    decoded_message = []

    with open(message_file, 'r') as file:
        lines = file.readlines()

        pyramid_words = [line.strip().split()[1] for line in lines]
        pyramid_ids = [int(line.strip().split()[0]) for line in lines]
        print(max(pyramid_ids))
        end_idx = 0
        length = len(pyramid_ids)

        result = []
        cnt = 2
        pyramids_confirm_flag = False
        while end_idx < length:
            index = pyramid_ids.index(end_idx + 1)
            result.append(pyramid_words[index])
            end_idx = end_idx +cnt
            cnt = cnt + 1
            if end_idx == length - 1:
                pyramids_confirm_flag = True
    if pyramids_confirm_flag:
        return ' '.join(result)
    else:
        return 'not pyramids satisfied'





# Example usage
decoded_message = decode("coding_qual_input.txt")
print(decoded_message)