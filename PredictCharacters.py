import SegmentCharacters
import TrainRecognizeCharacters
import numpy as np
print("Loading model")
outW_antrenat = np.load('./saved_data/outW_data1.npy')
inW_antrenat = np.load('./saved_data/inW_data1.npy')
print('Model loaded. Predicting characters of number plate')

classification_result = []
for each_character in SegmentCharacters.characters:
    # convertesc fiecare imagine 20x20 într-una 1x400
    each_character = each_character.reshape(1, -1);
    each_character = np.transpose(each_character)
    result = TrainRecognizeCharacters.elmPredict_optim(each_character, inW_antrenat, outW_antrenat, 2)
    classification_result.append(result)

print('Classification result')
plate_string= ''
for result in classification_result:
    index_max = np.argmax(result)
    print(TrainRecognizeCharacters.letters[index_max])
    plate_string+=str(TrainRecognizeCharacters.letters[index_max])

print('Predicted license plate')
print(plate_string)

# exista posibilitatea ca, caractere sa fie ordonate gresit
# de aceea variabila column_list va fi folosita pentru
# ordonarea acestora în ordinea corecta.

column_list_copy = SegmentCharacters.column_list[:]
SegmentCharacters.column_list.sort()
rightplate_string = ''
for each in SegmentCharacters.column_list:
    rightplate_string += plate_string[column_list_copy.index(each)]

print('License plate')
print(rightplate_string)