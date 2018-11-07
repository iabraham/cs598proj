file_object  = open("000annotations.txt", "r")
#print(file_object.read())
files = list()
objects = list()
for line in file_object:
    #print(line)
    words = line.split()
    #print(words)
    for word in words:
        word_split = word.split('.')
        for new_word in word_split:
            if new_word == 'jpg':
                files.append(word)
    if words[0] == 'Objects':
        objects_line = list()
        for word in words:
            if word != 'Objects' and word != 'Detected:':
                objects_line.append(word)
        objects.append(objects_line)
#print(files)
#print(objects)

file_object  = open("000annotations2.txt", "r")
files_SRGAN_downsampled = list()
objects_SRGAN_downsampled = list()
for line in file_object:
    #print(line)
    words = line.split()
    #print(words)
    for word in words:
        word_split = word.split('.')
        for new_word in word_split:
            if new_word == 'jpg':
                files_SRGAN_downsampled.append(word)
    if words[0] == 'Objects':
        objects_line = list()
        for word in words:
            if word != 'Objects' and word != 'Detected:':
                objects_line.append(word)
        objects_SRGAN_downsampled.append(objects_line)
#print(files)
#print(objects)

err = 0
for i, file in enumerate(files):
    for j, file_downsampled in enumerate(files_SRGAN_downsampled):
        if file_downsampled == file:
            for k, classes_real in enumerate(objects[i]):
                for h, classes_found in enumerate(objects_SRGAN_downsampled[j]):
                    if classes_real != classes_found and k == h:
                        err += 1
print(err)







