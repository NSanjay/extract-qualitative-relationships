# extract-qualitative-relationships
Extract elements of a qualitative relationship from sentences expressing these relationships in natural language.

## Qualitative Relationships
A qualitative relationship expresses the relationship between two physical concepts in natural language.

Example: `more pollutants mean poorer air quality.`

In the above example, the relationship between the amount of pollutants and air quality is expressed qualitatively. The pieces of a qualitative relationship here are:

1. concepts:
	- pollutants
	- air quality
2. intensifier (describing intensity of a concept)
	- more (more)
	- poorer (less)

Expected Output: `(more, pollutants) <-> (less, air quality)`. The values {more / less} in the parantheses indicate the relative amount (intensity) of the concepts. `Poorer air quality` indicates that air quality is less, `more pollutants` indicates that pollutants are more.

## Dataset
The dataset consists of knowledge sentences from the train, dev and test splits of the QuaRTz dataset, created by AllenAI.

## Outline of Approach
1. Use a generate-validate framework to generate possible templates of possible correct output pairs, and validate which template can be inferred from the knowledge sentence - K<sub>i</sub>.
2. Use Natural Language Inference for the validate part.

## Details of Approach
1. For each sentence containing a qualitative relationship, generate templates of the form:
	`{more/less}` NP<sub>1</sub> `leads to {more/less}` NP<sub>2</sub> Here NP<sub>1</sub> and NP<sub>2</sub> are noun phrases extracted from the sentence using Stanford CoreNLP. The assumption made here is that physical concepts are noun phrases. For each <NP<sub>1</sub>, NP<sub>2</sub>> extracted, each template is a permutation of the intensifiers of <NP<sub>1</sub>, NP<sub>2</sub>.
	Examples: 
	1. `more pollutants leads to more air quality`
	2. `less pollutants leads to less air quality`
	3. `less pollutants leads to more air quality`
	4. `less pollutants leads to more air quality`.

	Each combination of K<sub>i</sub> and a template T<sub>j</sub> is a different record in this case.

2. Use Natural Language Inference (BERT is used here) with K<sub>i</sub> as premise and T<sub>j</sub> as hypothesis. Retrieve the template with the highest entailment score: ```max(NLI(K_i, T_j)```.

3. Create another dataset such that (K<sub>i</sub>, T<sub>j</sub>) have label 1, and other templates (K<sub>i</sub>, T<sub>m</sub>), with (T<sub>m</sub> != T<sub>j</sub>) have label 0. Oversample label 1 examples to create a balanced dataset.

4. Train a NLI model using the above created dataset.

5. Measure accuracy on the test set. An example is correctly predicted if the template T<sub>j</sub> with label 1 has the highest entailment score.

## Installation
1. Download this repository.
2. Install `pytorch_transformers`. (UPDATE: this library is updated to transformers. Download the previous version.). Also install stanford core NLP. Run `Paraphrases.java` in `candidate_generation\src\main\java\stanford_test\candidate_generation\ParaPhrases.java`
3. Run `generate_choices.py`. Check the command line arguments to use with different files.
4. Run `score_candidate_sentences_with_para.py`.
5. Run `retrieve_top_hypotheses.py`.
6. Run `generate_candidates_dataset.py`.
7. Run `train_sequence_classification_model.py`.
8. Run `display_mcq_results.py`.
9. Run `display_results.py`.

## Accuracy
All split accuracies are 89%+.