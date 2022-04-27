import hashlib


class Trainer:
	@staticmethod
	def deterministic_float(string: str, seed='') -> float:
		return float(int(hashlib.sha256((string + seed).encode('utf-8')).hexdigest(), 16) % 2**16) / 2**16  # in [0, 1]

	@classmethod
	def split(cls, episodes, training_ratio=0.7, validation_ratio=0.15, seed=None):  # test_ratio is the rest
		seed = str(seed) if seed else ''
        
		training_set = []
		validation_set = []
		test_set = []
		for e in episodes:
			
			p = cls.deterministic_float(e['id'], seed)
			if p < training_ratio:
				training_set.append(e)
			elif p < training_ratio + validation_ratio:
				validation_set.append(e)
			else:
				test_set.append(e)

		return training_set, validation_set, test_set

	@classmethod
	def get_index(cls, action):
		if action['pose']['d'] < 0.04:
			return 0
		elif action['pose']['d'] < 0.06:
			return 1
		elif action['pose']['d'] < 0.08:
			return 2
		return 3
