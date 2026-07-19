import zipfile
import os
import re
import pandas as pd

ZIP_PATH = "../datasets/ami_corpus/transcripts_ami.corpus.zip"

EXTRACT_FOLDER = "../datasets/ami_corpus/extracted"

OUTPUT_FILE = "../output/ami_sentences.csv"

os.makedirs(EXTRACT_FOLDER, exist_ok=True)
os.makedirs("../output", exist_ok=True)

print("Extracting ZIP...")

with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
    zip_ref.extractall(EXTRACT_FOLDER)

print("Extraction Complete")

rows = []

sentence_id = 1

# Fillers to ignore
IGNORE = {

    "uh",
    "um",
    "umm",
    "hmm",
    "mm",
    "mmm",
    "yeah",
    "yep",
    "okay",
    "ok",
    "right",
    "kay",
    "'kay"

}

pattern = re.compile(
    r"\[(.*?)\]\s*(Speaker\s*[A-Z])\s*:\s*(.*)"
)

for root, _, files in os.walk(EXTRACT_FOLDER):

    for file in files:

        if not file.endswith(".txt"):
            continue

        transcript_id = file.replace(".txt","")

        filepath = os.path.join(root,file)

        with open(filepath,"r",encoding="utf-8",errors="ignore") as f:

            for line in f:

                line=line.strip()

                if not line:
                    continue

                match = pattern.match(line)

                if not match:
                    continue

                timestamp = match.group(1).strip()

                speaker = match.group(2).strip()

                sentence = match.group(3).strip()

                # remove extra spaces

                sentence = re.sub(r"\s+"," ",sentence)

                # remove fillers

                if sentence.lower().replace(".","") in IGNORE:
                    continue

                # ignore tiny sentences

                if len(sentence.split())<3:
                    continue

                rows.append({

                    "sentence_id":sentence_id,

                    "transcript_id":transcript_id,

                    "timestamp":timestamp,

                    "speaker":speaker,

                    "sentence":sentence

                })

                sentence_id+=1

df=pd.DataFrame(rows)

df.to_csv(OUTPUT_FILE,index=False)

print()

print(df.head())

print()

print("Total Clean Sentences :",len(df))

print()

print("Saved to :",OUTPUT_FILE)