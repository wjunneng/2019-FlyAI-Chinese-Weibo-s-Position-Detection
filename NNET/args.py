# -------------------------arg-------------------------
SOURCES_FILE = '../data/input/dev.csv'
ids_file = '../data/input/ids.txt'
answers_file = '../data/input/answers.txt'
questions_file = '../data/input/questions.txt'
labels_file = '../data/input/labels.txt'

# directory for input data
in_dir = '../data'
# directory for output pickles
out_dir = '../data/output'
# decide portion to spare for training and validation
portion = 0.8
# type of word embeddings baike
emb = "baike"
# baike_50 vector path
baike_dir = '../data/input/baike-50.vec.txt'
# all of labels
labels = ['AGAINST', 'NONE', 'FAVOR']
# use the all vocabulary
big_voc = False
# max time step of sentence sequence
sen_max_len = 50
# max time step of sentence sequence
ask_max_len = 25
