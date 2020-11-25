import argparse
import os
import json
import gpt_2_simple as gpt2
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--path-to-model-dir",
                    help="path to model directory",
                    type=str,
                    required=True,
                    dest="path_to_model_dir")

parser.add_argument("--path-to-params-dict",
                    help="path to params dict directory",
                    type=str,
                    required=True,
                    dest="path_to_params_dict")

# parser.add_argument("--path-to-memes-db",
#                     help="path to output file",
#                     type=str,
#                     required=True,
#                     dest="path_to_memes_db")

parser.add_argument("--output-path",
                    help="path to output file",
                    type=str,
                    required=True,
                    dest="output_path")

parser.add_argument("--n-samples",
                    help="number of samples for model to generate",
                    type=int,
                    required=True,
                    dest="n_samples")

parser.add_argument("--batch-size",
                    help="batch_size for model inference. n_samples % batch_size == 0 should be",
                    type=int,
                    required=True,
                    dest="batch_size")


# parser.add_argument("--write-mode",
#                     help="Write mode for writing output. Could be 'a' for append or 'w' for write",
#                     type=str,
#                     choices=["a", "w"],
#                     required=True,
#                     dest="write_mode")

args = parser.parse_args()

PATH_TO_MODEL = args.path_to_model_dir
print(PATH_TO_MODEL)
PATH_TO_PARAMS_DICT = args.path_to_params_dict
# PATH_TO_MEMES_DB = args.path_to_memes_db
OUTPUT_PATH = args.output_path
CHECKPOINT_DIR = PATH_TO_MODEL + "checkpoint/"
BATCH_SIZE = args.batch_size
N_SAMPLES = args.n_samples
RUN_NAME = "prod"
START_TOKEN = "<|startoftext|>"

sess = gpt2.start_tf_sess()
print("Start loading model...")
gpt2.load_gpt2(sess, checkpoint_dir=CHECKPOINT_DIR, run_name=RUN_NAME)
print("Model loaded")
with open(PATH_TO_PARAMS_DICT, "r") as fin:
    params_dict = json.load(fin)
print("Start generating...")
output = gpt2.generate(sess,
                       checkpoint_dir=CHECKPOINT_DIR,
                       run_name=RUN_NAME,
                       nsamples=N_SAMPLES,
                       batch_size=BATCH_SIZE,
                       return_as_list=True,
                       **params_dict)

res = [x.strip(START_TOKEN) for x in output]
res_df = pd.DataFrame(res, columns=["text"])

# res_df.to_csv(PATH_TO_MEMES_DB, mode="a", index=False)
res_df.to_csv(OUTPUT_PATH, mode="w", index=False)


# python inference.py --path-to-model-dir=.\disaster_girl\ --path-to-params-dict=.\disaster_girl\params_dict.json --output-path=.\disaster_girl\output_test.csv --path-to-memes-db=.\disaster_girl\memes_db.csv  --n-samples=100 --batch-size=20
