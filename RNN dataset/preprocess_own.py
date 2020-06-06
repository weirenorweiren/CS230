import codecs
import re
import numpy as np
import os
from random import shuffle

# We follow RNN paper's preprocess.py to process our dataset as required
source = 'name_nationality'
category = 'own'
data_dir = './data/' + source

unigram_set_path = './data/' + category + '/0_unigram_to_idx.txt'
bigram_set_path = './data/' + category + '/1_bigram_to_idx.txt'
trigram_set_path = './data/' + category + '/2_trigram_to_idx.txt'

fourgram_set_path = './data/' + category + '/3_fourgram_to_idx.txt'

country_set_path = './data/' + category + '/country_to_idx.txt'
new_dataset_path = './data/' + category + '/data_'

name_to_country = {} # Dictionary with names as keys and countries as values
unigram_set = set()
bigram_set = set()
trigram_set = set()

fourgram_set = set()

country_cnt = {}

clean = False # True for once then no need to clean again; actually since we don't want add '$', it can be set to False
write = True 
allowed_char = ['.', '\'']
cleaning_char = [':', '©', '¶']

# Go over each files in data_dir => './data/test'
for root, dir, files in os.walk(data_dir): 
	for file_cnt, file_name in enumerate(files):  
		file_path = data_dir + '/' + file_name
		names = codecs.open(file_path, 'r', 'utf-8').readlines() # A list of names
		country = file_name.split('_')[0]
		if country[-1] == '2':
			country = country[:-1]
		for name in names:
			name = name[:-1] # Remove '/n' from the end of each line
			if clean:
				# check for special characters 
				name = re.sub(r'[-–]', ' ', name)
				name = re.sub(r'\([^)]*\)', '', name)    
				name = re.sub(r'\"[^)]*\"', '', name)
				name = re.sub(r'\s\s', ' ', name)
				name = re.sub(r'[\s\t]$', '', name)
				name = re.sub(r'\d', '', name)
				name = name.lower()
				name = name.split(' ')
				name = ['$' + n + '$' if i < len(name) - 1 else '+' + n + '+' 
						for i, n in enumerate(name)]
				name = ' '.join(name)
				 
				assert not name.endswith(' ') and not name.startswith(' ')
				assert not '  ' in name, (name, 'has double space!')

				if any(c in cleaning_char for c in name):
					continue
			
			name_to_country[name] = country # Save the washed name into the corresponding dictionary

			if country not in country_cnt:
				country_cnt[country] = 0 # Introduce a new country name to count for the quantitiy of names for a certain country
			country_cnt[country] += 1

if write: # Write into new txt files as required by RNN paper
	data_to_write = {}
	train_data = {}
	valid_data = {}
	test_data = {}
	name_to_country = list(name_to_country.items()) # dict.items() => a list of tuples with the value from dictionary
	shuffle(name_to_country)
	for name, country in name_to_country:
		for char_idx, char in enumerate(name): # Create n-gram set
			unigram_set.add(char)
			if char_idx > 0:
				bigram_set.add(name[char_idx - 1] + name[char_idx])
			if char_idx > 1:
				trigram_set.add(name[char_idx - 2] + name[char_idx - 1] + name[char_idx])
			if char_idx > 2:
				fourgram_set.add(name[char_idx - 3] + name[char_idx - 2] + name[char_idx - 1] + name[char_idx])

		if country not in train_data.values(): # dict.values() => a list of values
			train_data[name] = country # Ensure each country appears in the training data
		else:
			data_to_write[name] = country # Data to distribute later (train, valid, test)
		
	data_size = len(train_data) +  len(data_to_write)
	# Total dataset size
	data_to_write = list(data_to_write.items()) # Change the type of data_to_write
	shuffle(data_to_write)
	for name, country in data_to_write: # Allocate the data into (train, valid, test) => 3:1:1
		if len(train_data) < data_size * 0.6:
			train_data[name] = country
		elif len(valid_data) < data_size * 0.2:
			valid_data[name] = country
		else:
			test_data[name] = country
    
    new_dataset = open(new_dataset_path + category + '_train', 'wb') # For Windows, r would change '\r\n' to '\n' and w would change '\n' to '\r\n'; solved by rb or wb; also not for Unix/Linux
    for name, country in train_data.items():
        line = name + '\t' + country + '\n'
        new_dataset.write(line.encode('utf-8'))
        # new_dataset.write(name + '\t' + country + '\n')
    new_dataset.close()
    new_dataset = open(new_dataset_path + category + '_valid', 'wb') # But if we write in binary form, we need to encode the input with utf-8
    for name, country in valid_data.items():
        line = name + '\t' + country + '\n'
        new_dataset.write(line.encode('utf-8'))
        # new_dataset.write(name + '\t' + country + '\n')
    new_dataset.close()
    new_dataset = open(new_dataset_path + category + '_test', 'wb') # No indent! Always space!
    for name, country in test_data.items():
        line = name + '\t' + country + '\n'
        new_dataset.write(line.encode('utf-8'))
        # new_dataset.write(name + '\t' + country + '\n')
    new_dataset.close()

	# new_dataset = open(new_dataset_path + category + '_train', 'w')
	# for name, country in train_data.items():
	# 	new_dataset.write(name + '\t' + country + '\n')
	# new_dataset.close()
	# new_dataset = open(new_dataset_path + category + '_valid', 'w')
	# for name, country in valid_data.items():
	# 	new_dataset.write(name + '\t' + country + '\n')
	# new_dataset.close()
	# new_dataset = open(new_dataset_path + category + '_test', 'w')
	# for name, country in test_data.items():
	# 	new_dataset.write(name + '\t' + country + '\n')
	# new_dataset.close()

	# Write unigram set
	unigram_dataset = open(unigram_set_path, 'wb')
	for idx, char in enumerate(sorted(unigram_set)):
		line = char + '\t' + str(idx) + '\n'
		unigram_dataset.write(line.encode('utf-8'))
	unigram_dataset.close()

	# Write bigram set
	bigram_dataset = open(bigram_set_path, 'wb')
	for idx, char in enumerate(sorted(bigram_set)):
		line = char + '\t' + str(idx) + '\n'
		bigram_dataset.write(line.encode('utf-8'))
	bigram_dataset.close()

	# Write trigram set
	trigram_dataset = open(trigram_set_path, 'wb')
	for idx, char in enumerate(sorted(trigram_set)):
		line = char + '\t' + str(idx) + '\n'
		trigram_dataset.write(line.encode('utf-8'))
	trigram_dataset.close()

    # write fourgram set
    fourgram_dataset = open(fourgram_set_path, 'wb')
    for idx, char in enumerate(sorted(fourgram_set)):
        line = char + '\t' + str(idx) + '\n'
        fourgram_dataset.write(line.encode('utf-8'))
    fourgram_dataset.close()

	# Write country set
	country_size = len(country_cnt)
	country_dataset = open(country_set_path, 'wb')
	write_idx = 0
	for idx, (country, cnt) in enumerate(sorted(country_cnt.items())):
		line = country + '\t' + str(idx) + '\n'
		if cnt < 5 and clean:
			country_size -= 1
			continue
		country_dataset.write(line.encode('utf-8'))
		write_idx += 1
	country_dataset.close()           

print('\ndataset size', data_size)
print('train test valid size', len(train_data), len(valid_data), len(test_data))
print('sample data', name_to_country[532], name_to_country[15])
print('unigram set', len(unigram_set))
print('bigram set', len(bigram_set))
print('trigram set', len(trigram_set))

print('fourgram set', len(fourgram_set))

print('country set', country_size) 