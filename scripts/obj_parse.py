file_object  = open("000annotations.txt", "r")
NUM_IMAGES = 450

#print(file_object.read())
files = list()
objects = list()
NUM_OBJECTS = 0
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
                NUM_OBJECTS += 1
        objects.append(objects_line)
#print(files)
#print(objects)

file_object  = open("object_out_all_blurs.txt", "r")
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

sinc_acc = 0.0
nearest_acc = 0.0
cubic_acc = 0.0
area_acc = 0.0
for i, file in enumerate(files):
    for j, file_downsampled in enumerate(files_SRGAN_downsampled):
        if file_downsampled == 'sinc' + file:
            for k, classes_real in enumerate(objects[i]):
                for h, classes_found in enumerate(objects_SRGAN_downsampled[j]):
                    if classes_found == classes_real:
                        sinc_acc += 1.0
                        objects_SRGAN_downsampled[j][h] = []
        elif file_downsampled == 'nearest' + file:
            for k, classes_real in enumerate(objects[i]):
                for h, classes_found in enumerate(objects_SRGAN_downsampled[j]):
                    if classes_found == classes_real:
                        nearest_acc += 1.0
                        objects_SRGAN_downsampled[j][h] = []
        elif file_downsampled == 'cubic' + file:
            for k, classes_real in enumerate(objects[i]):
                for h, classes_found in enumerate(objects_SRGAN_downsampled[j]):
                    if classes_found == classes_real:
                        cubic_acc += 1.0
                        objects_SRGAN_downsampled[j][h] = []
        elif file_downsampled == 'area' + file:
            for k, classes_real in enumerate(objects[i]):
                for h, classes_found in enumerate(objects_SRGAN_downsampled[j]):
                    if classes_found == classes_real:
                        area_acc += 1.0
                        objects_SRGAN_downsampled[j][h] = []
sinc_acc = sinc_acc / NUM_OBJECTS
nearest_acc = nearest_acc  / NUM_OBJECTS
cubic_acc = cubic_acc / NUM_OBJECTS
area_acc = area_acc / NUM_OBJECTS

print(sinc_acc)
print(nearest_acc)
print(cubic_acc)
print(area_acc)







