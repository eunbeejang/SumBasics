# SumBasics

Multi-document summarizer

Ani Nenkova and Lucy Vanderwende. The Impact of Frequency on Summarization. Microsoft Research, Redmond, Washington, Tech. Rep. MSR-TR-2005-101. 2005.

## Summary
Multi-document summarization is the task that creates a summary with the major points from multiple text resources of the same topic. The key aspect of a successful summarization system is the content selection which is used to find the important sentences for the summarization and the generation of output sentences. In the paper “ The Impact of Frequency on Summarization”, Nenkova et al. (2005) carry out empirical studies on how frequency-based feature design contributes to the performance of the summarizers by comparing the human and automatic summarizers. Moreover, the authors present SumBasic, a multi-document summarization system, which utilizes frequency of words and content units excessively, and show how their model outperforms several other existing systems.

Current automatic summarization models rely on identifying necessary sentences and extracting them. While a binary classifier, a Markov model, or the assignment of weights to text features for the feature selection is commonly used in the field, Nenkova et al. explore the significance of words and content frequency for the design of the model. When tested on 30 sets of 2003 DUC data, the scholars found that the words that are highly frequent from the input text tend to appear more in the human- generated summaries. Automatic summarizers showed a similar trend. However, the authors observed that a large percentage of less frequent words also appeared in the models, suggesting that it is crucial to identify the low-frequency words that play significant roles in the performance of the summarization systems.

In order to isolate the contribution of frequency information, Nenkova et al. introduce a summarization model called SumBasic, which greedily searches for important sentences based on frequency-based weights. It finds the sentences with highest probability score, where the score is computed by averaging the probability of word frequency in each sentence. Then the words from selected sentences get their frequency weights updated in order to address redundancy. The updating of the weights ensures the sensitivity to context and allows low scored words to have significance on the choice of subsequent sentences while avoiding a repetitive summary. The evaluation of the model is done on 50 test sets of DUC 2004 and 25 test sets of MSE 2005 datasets using the train set from DUC 2003. The system performed significantly better than 12 other systems when tested on DUC 2004 using ROUGE-1. On the other hand, the testing on MSE 2005 dataset with pyramid score, ROUGE-2 and R-SU4 did not show significant difference between the models tested. Moreover, Nenkova et al. studied the impact of duplication removal methods on SumBasic, LexRank, and DEMS. They found that SumBasic performed significantly better than the others when duplications are removed and reranked.


#### Summarizer Versions
1. **original**: The original version, including the non-redundancy update of the word scores.
2. **best-avg**: A version of the system that picks the sentence that has the highest average probability
in Step 2, skipping Step 3.
3. **simplified**: A simplified version of the system that holds the word scores constant and does not incorporate the non-redundancy update.
4. **leading**: The version which takes the leading sentences of one of the articles, up until the word length limit is reached.

#### To run
```
# python sumbasic.py <method_name> <file_n>*

python ./sumbasic.py simplified ./docs/doc1-*.txt > simplified-1.txt
```
