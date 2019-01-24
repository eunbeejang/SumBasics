
# Import Statements
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from functools import reduce
import argparse




class Preprocessor(object):


	def __init__(self):

		pass


	def sent_seg(self, doc_path):

		# Segment sentences
		with open(doc_path) as f:
			return sent_tokenize(f.read())


	def preprocess_sent(self, sent):

		# Tokenization and Lowercasing
		words = word_tokenize(sent.lower())

		# POS tagging
		words_pos = nltk.pos_tag(words)

		# Lemmantization
		lm_words = []
		lem = WordNetLemmatizer()
		for w, p in words_pos:
			#print(w, p)
			try:
				lm_words.append(lem.lemmatize(w, pos=p[0].lower()))
			except KeyError:
				lm_words.append(lem.lemmatize(w))

		# Remove stop words
		STOPS = stopwords.words('english')
		#print(lm_words)
		return [w for w in lm_words if w not in STOPS]


	def preprocess(self, doc_path):

		# Preprocess documents
		with open(doc_path) as f:
			return [self.preprocess_sent(sent) for sent in sent_tokenize(f.read())]


	def word_count(sent):

		if type(sent) is list:
			return sum(len(t.split()) for t in sent)

		return len(sent.split())





class Summarizer(object):


	def __init__(self, max_word_length=100):
		self.max_word_length = 100
		self.preprocessor = Preprocessor()


	def sumbasic_summarize(file_path, max_word_length, preprocessor, update):
		# SumBasic Summarizer
		"""
		if update == True:
			Original SumBasics(incorporates non-redundancy update of the word scores)
		if update == False:
			Simplified SumBasics(does not incorporate the non-redundancy update)
		"""

		sents_processed = [s for p in file_path for s in preprocessor.preprocess(p)]
		#print(sents_processed)
		count = Counter()
		for sent in sents_processed:
			count += Counter(sent)

		num_tok = float(sum(count.values()))

		probs = {w: count / num_tok for w, count in count.items()}

		sentences = [s for p in file_path for s in preprocessor.sent_seg(p)]

		summary = []


		# Add summary sentences
		while len(sentences) > 0 and Preprocessor.word_count(summary) < max_word_length:

			max_prob = float(0)
			max_sent = None

			for i, sent in enumerate(sentences):

				prob = reduce(lambda x, y: x * y, [probs[w] for w in sents_processed[i]])

				if max_prob < prob:
					max_prob = prob
					max_sent = sent

			summary.append(max_sent)
			sentences.remove(max_sent)

			# Update weights
			if update:
				for w in sents_processed[i]:
					probs[w] = probs[w] ** 2

		if Preprocessor.word_count(summary) > max_word_length:
			return ' '.join(summary[:-1])

		return ' '.join(summary)


	def sumbasic_best_avg_summarize(file_path, max_word_length, preprocessor):
		# SumBasic best-avg Summarizer
		"""
		if update == True:
			Original SumBasics(incorporates non-redundancy update of the word scores)
		if update == False:
			Simplified SumBasics(does not incorporate the non-redundancy update)
		"""
		sents_processed = [s for p in file_path for s in preprocessor.preprocess(p)]

		count = Counter()
		for sent in sents_processed:
			count += Counter(sent)

		num_tok = float(sum(count.values()))

		probs = {w: count / num_tok for w, count in count.items()}

		sentences = [s for p in file_path for s in preprocessor.sent_seg(p)]

		summary = []


		# Add summary sentences
		while len(sentences) > 0 and Preprocessor.word_count(summary) < max_word_length:

			max_prob = float(0)
			max_sent = None

			for i, sent in enumerate(sentences):

				prob = reduce(lambda x, y: x + y, [probs[w] for w in sents_processed[i]])
				prob = prob/len(sents_processed[i])

				if max_prob < prob:
					max_prob = prob
					max_sent = sent

			summary.append(max_sent)
			sentences.remove(max_sent)

			# Update weights
			for w in sents_processed[i]:
				probs[w] = probs[w] ** 2

		if Preprocessor.word_count(summary) > max_word_length:
			return ' '.join(summary[:-1])

		return ' '.join(summary)


	def summarize(self, file_path):
		raise NotImplementedError

 



class Original(Summarizer):
	# The original version, including the non-redundancy update of the word scores.
	def summarize(self, file_path):
		return Summarizer.sumbasic_summarize(file_path, self.max_word_length, self.preprocessor, update=True)


class BestAvg(Summarizer):
	# A version of the system that picks the sentence that has the highest average probability in Step 2, skipping Step 3.
	def summarize(self, file_path):
		return Summarizer.sumbasic_best_avg_summarize(file_path, self.max_word_length, self.preprocessor)


class Simplified(Summarizer):
	# A simplified version of the system that holds the word scores constant and does not incorporate the non-redundancy update.
	def summarize(self, file_path):
		return Summarizer.sumbasic_summarize(file_path, self.max_word_length, self.preprocessor, update=False)


class Leading(Summarizer):
	#  A baseline summarizer which takes the leading sentences of one of the articles, up until the word length limit is reached

	def summarize(self, doc_paths):
		max_word_length = 100

		sentences = self.preprocessor.sent_seg(doc_paths[0])

		if sum(Preprocessor.word_count(sent) for sent in sentences) <= self.max_word_length:
			return ' '.join(sentences)

		summary, i = [], 0

		while Preprocessor.word_count(summary) <= self.max_word_length:
			summary.append(sentences[i])
			i += 1

		return ' '.join(summary[:-1])



def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('which_summarizer', help="Method name ('leading', 'best-avg','simplified', or 'original')")
	parser.add_argument('article_num', help="Filepaths to documents to summarize", nargs='+')
	args = parser.parse_args()

	if args.which_summarizer == 'original':
		summarizer = Original()
	elif args.which_summarizer == 'best-avg':
		summarizer = BestAvg()
	elif args.which_summarizer == 'simplified':
		summarizer = Simplified()
	elif args.which_summarizer == 'leading':
		summarizer = Leading()
	else:
		ValueError("method_name must be 'original', 'best-avg', 'simplified', or 'leading")

	print (summarizer.summarize(args.article_num))


if __name__ == '__main__':
	main()