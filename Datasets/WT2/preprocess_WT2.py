import numpy as np

def load_wt2():
	import sentencepiece as spm
	import numpy as np
	
	sp = spm.SentencePieceProcessor()
	sp.load('lambada.model')
	predefined_tokens = ['<eod>']
	replace = [(' @-@ ',
				'-'),
			   (' @.@ ',
				'.'),
			   (' @,@ ',
				','),
			   ('<unk>',
				'<')]
	files = ['WT2.test.txt',
			 'WT2.train.txt',
			 'WT2.valid.txt']
	tasks = ['test',
			 'train',
			 'valid']
	data = {'train': np.zeros([74025,
							   275]),
			'test': np.zeros([8982, 
							  176]),
			'valid': np.zeros([7830,
							   243])}
	for file in range(len(files)):
		task = tasks[file]
		txt_file = open(files[file],
						'r').readlines()
		line_iteration = 0
		for line in txt_file:
			if (line == '<new> \n'):
				data[task][line_iteration,0] = 1
			else:
				for a, b in replace:
					line = line.replace(a,
										b)
				tkn_array = np.array(sp.encode_as_ids(line)) + 2
				for sequence in range(len(tkn_array)):
					data[task][line_iteration,sequence] = tkn_array[sequence]
			line_iteration += 1
	return data['train'], data['test'], data['valid']

if __name__ == '__main__':
	train, test, valid = load_wt2()
	
	np.save('WT2.train.npy',
			train)
	np.save('WT2.test.npy',
			test)
	np.save('WT2.valid.npy',
			valid)
