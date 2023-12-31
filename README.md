# NMT Augmentation: Preserving Untranslatable Text Elements with Do-Not-Translate Tags

In the context of Neural Machine Translation (NMT) projects, a common challenge arises when certain parts of the text should not be translated. These parts could include names, alphanumeric codes, or larger blocks of text that must remain unchanged during the translation process. Traditional Transformer architectures used in NMT projects do not handle this requirement directly.

This repository addresses this challenge by providing a solution to create specific parallel corpora for training NMT engines with entity-augmented data. The goal is to train the engine to recognize and retain specific "Do-Not-Translate" (DNT) tags during inference.

In a typical scenario, you have test data (a document) with specific portions that should not be translated. These portions are manually identified, converted to tags, and then used by this repository to augment the training data. During translation, the engine is expected to return these tags, allowing the mapping back to the original entities.

The process maintains the natural sense at the sentence level. Matching entities in the training parallel corpora are replaced with a special token `${DNT0}X`, where the prefix is a tag, and `X` is an arbitrary number for enumeration. This allows the NMT engine to learn to leave these entities untouched during inference on test data.

The pipeline involves using various tools for aligning and tagging sentences, resulting in a parallel dataset with ${DNT0}X tags replacing selected entities. It's worth noting that while this pipeline is well-suited for languages with Latin, Arabic, and Cyrillic alphabets, challenges may arise with languages using different alphabets, as discussed in detail [`HERE`](https://github.com/Dpm-a/Corpora-DNTs-Augmenting/blob/main/MasterThesis.pdf).

# Documentation

It is recommended installing a fresh Conda environment and run python 3.8.

```bash
conda create --name <my_env> python==3.8
conda activate <my_env>
```

Once activated, we need to install dependencies:

```bash
git clone https://github.com/Dpm-a/DNTs
cd DNTs/
pip install -r requirements.txt
```

# I. DeepPavlov

This library is used for a big variety of tasks, including Naming Entity Recognition which is in line with our goals.

DeepPavlov has been chosen due to the vast availability of models, including the **ner_ontonotes_bert_mult** multilingual one that we are going to exploit (more on [OntoNotes](https://paperswithcode.com/dataset/ontonotes-5-0)).

First, let's move in the repo's folder:

```bash
cd DNTs/src
```

In order to tag our text using DeepPavlov library we are going to use the following file which will evenually return two binary `.pickle` files containing, once decoded, lists of tuple of the kind `[(word, tag), …, (word, tag)]` for each input/target sentence:

```bash
python pavlov_tagger_pickle.py -s <source.file> -t <target.file>
```

The source and target texts are **tokenized** and **tagged**. For each token, there is a tag with **BIO markup**. Tags are separated from tokens with whitespaces. Finally, Sentences are separated with empty lines.

### GPU usage for pavlov

**!NOTE**: Run the script on machine with a decent GPU (Nvidia Quadro T4 or better is best), the NER tagger is based on BERT Transformer and a gpu is a good boost for inference, considering corpora of millions of rows.
More on BERT architecture:

- [Jacob Devlin](https://arxiv.org/search/cs?searchtype=author&query=Devlin%2C+J), [Ming-Wei Chang](https://dblp.uni-trier.de/pid/69/4618.html), [Kenton Lee](https://dblp.uni-trier.de/pid/121/7560.html) and [Kristina Toutanova](https://dblp.uni-trier.de/pid/25/1520.html?q=Kristina%20Toutanova). (2019). [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

To run or train PyTorch-based DeepPavlov models on GPU you should have [CUDA](https://developer.nvidia.com/cuda-toolkit) installed on your host machine, and install model’s package requirements. CUDA version should be compatible with DeepPavlov [required PyTorch version](https://github.com/Dpm-a/DNTs/blob/main/requirements.txt).

Let's say you want to use your #3 device (gpu), make sure to make the cude device visible for the script by either running CLI command:

```bash
  export CUDA_VISIBLE_DEVICES=3; python -m deeppavlov train <config_path>
```

or, in python:

```python
import os

os.environ["CUDA_VISIBLE_DEVICES"]="3"
```

For anything else, please give a look to the official documentation of [DeepPavlov](https://docs.deeppavlov.ai/en/master/).

# II. FastAlign

Once the tagged text is output we are going to exploit Fastalign to generate alings between each pair of translation. We are going to take into account aliments made by both sides (from left sentence to right one and viceversa), leading us to a more concise union of the two sets.

Building `fast_align` requires a modern C++ compiler and the CMake build system. Additionally, the following libraries can be used to obtain better performance.

To compile and run it, do the following:

```bash
git clone https://github.com/clab/fast_align
cd fast_align
mkdir build
cd build
cmake ..
make
```

**Note**: DeepPavlov will generate tokenized text, and fastalign will need those tokens to be aligned (otherwise the plain original string will produce less alignments with respect to real tags).<br>
For this reason we are going to use the following commands in `build/` folder:

```bash
python merge_and_fast_align.py -s source.pavlov -t target.pavlov -i 5
```

where:

- **-i** → Fast Align iterations (int → [0,inf])
- Decodes our. pickles
- Merges each row in a parallel sentence in a fastalign readable format
  - (i.e. ‘Sentence_1 ||| Sentence_2’)
- Then it calls Fast Align CLI commands:
  - `bash ./fast_align -i merged_file -d -o -v -I 5 > forward.align`
  - `bash ./fast_align -i merged_file -d -o -v -r -I 5 > reverse.align`
  - `bash ./atools -i forward.align -j reverse.align -c union > symm.union.align`

You can run `fast_align` to see a list of command line options.

More on this software can be found on the original repo's page [Fast Align](https://github.com/clab/fast_align) or in the official paper:

- [Chris Dyer](http://www.cs.cmu.edu/~cdyer), [Victor Chahuneau](http://victor.chahuneau.fr), and [Noah A. Smith](http://www.cs.cmu.edu/~nasmith). (2013). [A Simple, Fast, and Effective Reparameterization of IBM Model 2](http://www.ark.cs.cmu.edu/cdyer/fast_valign.pdf). In _Proc. of NAACL_.

# III. DNT Augmentation

Finally, to replace DNT’s in the given corpora, run [`make_dnts.py`](https://github.com/Dpm-a/DNTs/blob/main/make_dnts_algorithm3.py):

```bash
python make_dnts.py -s <src>.pavlov -t <trg>.pavlov -a alignments.txt -g .5 -v 1
```

where:

- **"-s"**, **"--source"** → source file path
- **"-t"**, **"--target"** → target file path
- **"-a"**, **"--alignents"** → alignments file path
- **"-p"**, **"--probability"** → probability for each DNT to be replaced (float → [0,1])
- **"-g"**, **"--augment"** → probability for each DNT to be doubled (float → [0,1])
- **"-v"**, **"--verbosity"** → Verbosity, creates two more files (int → [0,1]):
  - `alignments.log` : alignments from Fast Align

This script will generate two files containing DNTs reaplacing entities and optionally a log to check alignments found and replaced by the script itself.

## Utils

[Utils folder](https://github.com/Dpm-a/DNTs/tree/main/utils) contains few additional scripts:

- [`shuffle_corpora.py`](https://github.com/Dpm-a/DNTs/blob/main/utils/shuffle_corpora.py),
  As the name suggests, it shuffles **in parallel** corpora mantaining indexes and thus the coupling of the sentences.<br>
  Use this script before feeding the training parallel corpora to your NMT model.<br>
  Usage:

  ```bash
  python suffle_corpora.py -s <source.file> -t <target.file>
  ```

  Optionally, and more RAM-friendly, only on Linux:

  ```bash
  # EXAMPLE on zipped files
  paste <(unxz -dc OPUS-News-Commentary_<src>-<trg>.<src>.xz) \
   <(unxz -dc OPUS-News-Commentary_<src>-<trg>.<src>.<trg>.xz) \
   | shuf > /tmp/__tmp__

  # get back original files
  cut -f1 /tmp/__tmp__ > /tmp/__tmp__.<src>
  cut -f2 /tmp/__tmp__ > /tmp/__tmp__.<trg>
  ```

- [`count_dnt.py`](https://github.com/Dpm-a/DNTs/blob/main/utils/count_dnt.py), counts DNTs tags inside both corpora, asserting the correctness of the process.<br>
  Usage:

  ```bash
  python count_dnt.py <source.file> <target.file>
  ```

  Optionally, only on linux:

  ```bash
  cat train.<src>-<trg>.<src> | grep -o "{DNT0}" | wc -l
  cat train.<src>-<trg>.<trg> | grep -o "{DNT0}" | wc -l
  ```

- [`check_dnt.py`](https://github.com/Dpm-a/DNTs/blob/main/utils/check_dnt.py), which provides some useful statistics on translations made by NMTs models. It firsts points out disalignments with respect to DNTs tag row by row, then calculates Precision and Recall on those results.<br>
  Usage:
  ```bash
  python check_dnt.py --ref <source.file> --hyp <target.file>
  ```
