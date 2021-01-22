+++
title="NER using OpenNLP. Data Preparation."
date=2020-11-10

[taxonomies]
categories = ["NLP", "Data Science"]
tags = ["opennlp","corpus annotation","model customization"]

[extra]
toc = true

+++

## Why NLP?

 Natural language processing is a component of Text Mining that performs a special kind of linguistic analysis that essentially helps a machine
to __read__ text. So this part of Data Science provides methods and approaches to mining patterns in texts. Thus NER is just one in the list of NLP tasks.
<!-- more -->
NER has a lot of synonyms, such as Entity Identification, Entity Chunking and Entity Extraction. 

**Named Entity Recognition** is a process of finding named-entities (entities, valuable for some special domain) in the text.

Standard kinds of entities looking for (using NER) in texts are:
- Person.
- Organization.
- Location.

__Let's clarify some common terms used in NLP:__

* Gazetteers (List of Seeds, Seeds, Named Entity Lists): list of domain entities. 
* Corpus (Corpora): a huge natural language text representing special domain.
* Annotation (Markup process): process of entities marking-up in a corpus.
* Learning process: training/testing process using annotated corpus.

# NER via OpenNLP library

We can use [Apache OpenNLP](https://opennlp.apache.org/) library to implement above process.

__Apache OpenNLP has a standard pipeline for NLP learning process:__
1) Preparing List of Entities
2) Annotating corpus / Use OpenNLP corpus
3) Training OpenNLP Parameters
4) Training own model / Use pre-trained models for corresponding corpus
5) Testing model with relevant data

## 1. Named-entity list preparations

There are a lot of examples how to use Apache OpenNLP standard domain with models by default. I would like to make the task more complex. In this case, let us
customize NER for medical domain and allow model to find medical entities in text documents.
We will try to learn model to find special entities in German language. There are no pretrained German models for Apache OpenNLP.

For the construction of a list of medical entities,  I used [MeSH open-sourced database](https://www.dimdi.de/dynamic/de/klassifikationen/weitere-klassifikationen-und-standards/mesh/index.html).
Medical Subject Headings (MeSH) DB – is a poly-hierarchical, concept-based thesaurus for the cataloging of book and media collections, indexing of databases and creation of search profiles in medical domain. Terms are extracted in the separate file where each line contains just one seed.

## 2. Corpus preparation

Corpus is important step of NER. Let us prepare an unlabelled corpus. Apache OpenNLP has a strict syntax for that. 

Corpus requirements:

- have lots of data. In OpenNLP documentation, it is recommended a minimum of 15000 sentences.
- entities should be surrounded by tags in format \<START:entity-name\> Entity \<END\>.  Entity name is a label of entities your learning model is going to look for.
- add spaces between tags (START/END) and labeled data.
- if possible, use one label per model. Multiple labels are possible, but it is not recommended.
- each line contains just one sentence. 
- empty lines delimit documents.  

Next steps 3-5 will be described in the next article.

Project source code using OpenNLP you can found here: [https://github.com/nnovakova/opennlp-annotator](https://github.com/nnovakova/opennlp-annotator)