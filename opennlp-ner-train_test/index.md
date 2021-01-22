+++
title="NER using OpenNLP. Custom Model Training and Validation."
date=2020-11-10

[taxonomies]
categories = ["Programming"]
tags = ["java","opennlp","learn model","model customization","train model","validate model"]

[extra]
toc = true

+++
NER using OpenNLP. Custom Model Training and Validation.

In Previous article I described how to prepare data (corpus) for training with OpenNLP in specific domain and custom entities (see https://nnovakova.github.io/opennlp-ner-annotaton/). 

Now it is time to use annotated corpus on the next stage of NLP pipeline.
Before we considered just the first step of pipeline – training data preparation. Now it is time to start real learning process.

But before we starting, let's list basic <b>OpenNLP conditions</b>:
Have enough data  (starts from 15 000 sentences in corpus).
- Corpus have to be annotated.
- Corpus must contain single language text.
- For API usage have to apply Java. 
- If these conditions are met, then you can start training.

<b>3. Training a model:</b>
<b>3.1 Read the training data </b>
        final MarkableFileInputStreamFactory in = new MarkableFileInputStreamFactory(new File(filePath));
       ObjectStream<NameSample> sampleStream
 = new NameSampleDataStream(new PlainTextByLineStream(in, StandardCharsets.UTF_8));

<b>3.2 Training Parameters </b>
final TrainingParameters params = new TrainingParameters();
params.put(TrainingParameters.ITERATIONS_PARAM, 100);// number of training iterations
params.put(TrainingParameters.CUTOFF_PARAM, 3);//minimal number of times a feature must be seen
params.put(TrainingParameters.ALGORITHM_PARAM, ModelType.MAXENT.toString());
<b>3.3 Train the model</b>
TokenNameFinderFactory nameFinderFactory = TokenNameFinderFactory.create(null, null, Collections.emptyMap(), new BioCodec());
String type = null; // null value - to use names from corpus.
model = NameFinderME.train("de", type, sampleStream, params, nameFinderFactory);
TokenNameFinderFactory nameFinderFactory = TokenNameFinderFactory.create(null, null, Collections.emptyMap(), new BioCodec());
String type = null; // null value - to use names from corpus.
model = NameFinderME.train("de", type, sampleStream, params, nameFinderFactory);
<b>3.4 Save the model to a file</b>
TokenNameFinderModel model = trainModel();
try (BufferedOutputStream modelOut = new BufferedOutputStream(new FileOutputStream(onlpModelPath))) {
    model.serialize(modelOut);
}
<b>4. Validate Mode</b>
For validation it is needed to load model from bin file.
try (InputStream modelIn = new FileInputStream(Config.onlpModelPath)) {
    TokenNameFinderModel model = new TokenNameFinderModel(modelIn);
    NameFinderME nameFinder = new NameFinderME(model);
    String sentence = ""Obwohl die erstmalige Isolierung von H5N1 bei Menschen schon im Jahr 1997\n" +
    "erfolgte, hat das Virus den letzten Schritt zu einem Pandemieauslöser bisher nicht\n" +
    "getan.\n";
    sentence = sentence.replaceAll("(?!\\s)\\W", " $0");
    String[] s = sentence.split(" ");
    // Get NE in sentence and positions of found NE
    Span[] nameSpans = nameFinder.find(s);
    System.out.println(Arrays.toString(nameSpans));
    String ent;
    for (Span nameSpan : nameSpans) {
        int entityLength = nameSpan.getEnd() - nameSpan.getStart();
        for (int i = 1; i <= entityLength; i++) {
            ent = sentence.split(" ")[nameSpan.getStart()];
            System.out.println(ent);
        }
    }
}
As a result you'll get a serial number of entities in the text + entity type name and a list of entities:
[[3..4) med-entity, [5..6) med-entity, [7..8) med-entity, [15..16) med-entity, [21..23) med-entity]
Isolierung
H5N1
Menschen
Virus
Pandemieausl

<b>Summary: </b>
OpenNLP is a good free of charge alternative for NER task solving. The biggest problem of using OpenNLP is a corpus preparation. It takes a lot of resources and required accurate annotation. This accuracy directly affects the training quality.

