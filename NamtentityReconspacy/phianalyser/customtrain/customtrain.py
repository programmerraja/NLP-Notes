import spacy
from spacy.util import minibatch, compounding
from spacy.training import Example

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

def train_ner_v3(TRAIN_DATA):
    nlp = spacy.load("en_core_web_lg")
    
    ner = nlp.get_pipe("ner")
    for _, ann in TRAIN_DATA:
        for _, _, label in ann["entities"]:
            ner.add_label(label)
    
    # CORRECT: create_optimizer() for PRETRAINED models
    optimizer = nlp.create_optimizer()
    
    other_pipes = [p for p in nlp.pipe_names if p != "ner"]
    with nlp.disable_pipes(*other_pipes):
        for itn in range(20):
            losses = {}
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                examples = [Example.from_dict(nlp.make_doc(text), ann) for text, ann in batch]
                nlp.update(examples, drop=0.35, sgd=optimizer, losses=losses)
            print(f"Iteration {itn+1}: {losses}")
    return nlp
nlp = train_ner_v3(TRAIN_DATA)
doc = nlp("My ID is three hundred twenty nine zero zero.")
print([(ent.text, ent.label_) for ent in doc.ents])
