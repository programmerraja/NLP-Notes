import spacy
from spacy.training.example import Example
from spacy.util import minibatch
import random
import spacy.cli.train as train_cli
from pathlib import Path

# Load en_core_web_lg
nlp = spacy.load("en_core_web_lg")


TRAIN_DATA = TRAIN_DATA = [
    # NAME (patient names)
    (
        "Patient name is Mariya Johnson. Family member is John Johnson.",
        {"entities": [(13, 26, "NAME"), (43, 55, "NAME")]},
    ),
    (
        "Andre, spelled A-N-D-R-E, is the patient.",
        {"entities": [(0, 43, "NAME")]},
    ),
    # DOB (dates of birth)
    (
        "Date of birth April eight, nineteen fifty-seven.",
        {"entities": [(14, 47, "DOB")]},
    ),
    (
        "DOB 4-80-57 or September fifth, nineteen fifty-eight.",
        {"entities": [(4, 10, "DOB"), (14, 46, "DOB")]},
    ),
    # PHONE (patient phone numbers)
    (
        "Your phone number seven six zero four oh four oh three seven zero.",
        {"entities": [(18, 60, "PHONE")]},
    ),
    (
        "Call me at 424-439-2100 or (760)927-9499.",
        {"entities": [(13, 24, "PHONE"), (28, 41, "PHONE")]},
    ),
    # ADDRESS (patient addresses)
    (
        "Address 1668 Taylor Fuse Road, apartment 1329688, Chennai, TN 600001.",
        {"entities": [(8, 56, "ADDRESS")]},
    ),
    (
        "1668 Fuse Road, no, 1668 Taylor Fuse Road, zip 600001.",
        {"entities": [(0, 47, "ADDRESS")]},
    ),
    # INSURANCE (insurance details)
    (
        "Insurance Blue Cross Aetna plan member ID ABC123.",
        {"entities": [(10, 20, "INSURANCE"), (21, 26, "INSURANCE")]},
    ),
    # ID (member IDs, policy numbers, etc.)
    (
        "Member ID ICA082206, policy number 123456789, claim 987654.",
        {"entities": [(10, 20, "ID"), (30, 41, "ID"), (52, 59, "ID")]},
    ),
    (
        "India Charlie Alpha 08-2206 is the group number.",
        {"entities": [(0, 23, "ID")]},
    ),
    # EMAIL
    (
        "Email patient@example.com or family@domain.net.",
        {"entities": [(6, 23, "EMAIL"), (28, 42, "EMAIL")]},
    ),
    # Mixed examples (keep facility names, doctors, etc.)
    (
        "Patient [NAME_1] visited Dr. Gupta at Sol dummy, call us at 760-728-1900.",
        {"entities": [(8, 15, "NAME")]},
    ),
    (
        "Your DOB [DOB_1], address [ADDRESS_1], insurance Blue Cross.",
        {"entities": [(9, 15, "DOB"), (24, 34, "ADDRESS"), (45, 55, "INSURANCE")]},
    ),
]

# Get NER pipe or add if missing
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner")
else:
    ner = nlp.get_pipe("ner")

# Add custom labels
for _, annotations in TRAIN_DATA:
    for ent in annotations.get("entities", []):
        ner.add_label(ent[2])

# Disable other pipes for training
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
with nlp.disable_pipes(*other_pipes):  # Train only NER
    optimizer = nlp.begin_training()

    for i in range(20):  # Iterations
        random.shuffle(TRAIN_DATA)
        losses = {}
        batches = minibatch(TRAIN_DATA, size=2)
        for batch in batches:
            examples = []
            for text, annotations in batch:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                examples.append(example)
            nlp.update(examples, sgd=optimizer, losses=losses)
        print(f"Iteration {i}: {losses}")

# Save model
nlp.to_disk("./custom_ner_model")
