file_object  = open("000annotations.txt", "r")
#print(file_object.read())
files = list()
objects = list()
for line in file_object:
    print(line)
    words = line.split()
    print(words)
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
print(files)
print(objects)








