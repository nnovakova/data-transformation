+++
title="NER using OpenNLP. Model Training and Validation."
date=2020-11-17

[taxonomies]
categories = ["Programming"]
tags = ["java","opennlp","learn model","model customization","train model","validate model"]

[extra]
toc = true

+++

In [the previous article](https://nnovakova.github.io/opennlp-ner-annotaton/) I described how to prepare training data (corpus) for training with OpenNLP for a specific domain and custom entities. 
<!-- more -->

Now it is time to use annotated corpus for the next stage of NLP pipeline.
Before, we completed the first step of the pipeline – training data preparation. Now it is time to start real learning process.

But before we start next step, let's list basic __OpenNLP conditions__:

- Have enough data  (starts from 15 000 sentences in corpus).
- Corpus must to be annotated.
- Corpus must contain single language text.
- Write Java code to use OpenNLP API. 

If these conditions are met, then you can start training.

# 3. Training a model

Below, in all code snippets I am using Java 11 that allows to use `var` keywords for local variables.

## 3.1 Read the training data

```java
var in = new MarkableFileInputStreamFactory(new File(filePath));
var sampleStream = new NameSampleDataStream(new PlainTextByLineStream(in, StandardCharsets.UTF_8));
```

## 3.2 Training Parameters

```java
var params = new TrainingParameters();
params.put(TrainingParameters.ITERATIONS_PARAM, 100);// number of training iterations
params.put(TrainingParameters.CUTOFF_PARAM, 3);//minimal number of times a feature must be seen
params.put(TrainingParameters.ALGORITHM_PARAM, ModelType.MAXENT.toString());
```

## 3.3 Train the model

```java
var nameFinderFactory = TokenNameFinderFactory.create(
    null, null, Collections.emptyMap(), new BioCodec());
String type = null; // null - to use entity names from the corpus annotations
model = NameFinderME.train("de", type, sampleStream, params, nameFinderFactory);
```

## 3.4 Save the model to a file

```java
try (BufferedOutputStream modelOut = new BufferedOutputStream(new FileOutputStream(onlpModelPath))) {
    model.serialize(modelOut);
}
```

# 4. Validate Mode

We need to load the model from the binary file to validate it:

```java
try (InputStream modelIn = new FileInputStream(Config.onlpModelPath)) {
    var model = new TokenNameFinderModel(modelIn);
    var nameFinder = new NameFinderME(model);
    var sentence = "Keine Mittellinienverlagerung.\n" +
            "Keine intrakranielle Blutung.\n" +
            "Kein klares fokales, kein generalisiertes Hirnödem.\n" +
            "Derzeit keine eindeutige frische, keine grobe alte Ischämie.";
    ;
    sentence = sentence.replaceAll("(?!\\s)\\W", " $0");
    var words = sentence.split(" ");

    // Get NE in sentence and positions of found NE
    var nameSpans = nameFinder.find(words);
    System.out.println(Arrays.toString(nameSpans));

    for (Span span : nameSpans) {
        var ent = words[span.getStart()];
        System.out.println(ent);
    }
}
```

As a result you'll get a serial number of entities in the text + entity type name and a list of entities:

```bash
[[3..4) med-entity, [5..6) med-entity, [7..8) med-entity, [15..16) med-entity, [21..23) med-entity]
Isolierung
H5N1
Menschen
Virus
Pandemieausl
```

# Summary

OpenNLP is a robust and free open source alternative for NER task solving. 
The biggest problem when using OpenNLP is a corpus preparation. It takes a lot of resources and requires accurate annotation. This accuracy directly affects the training quality.

Project source code using OpenNLP you can found here: [https://github.com/nnovakova/opennlp-ner](https://github.com/nnovakova/opennlp-ner)