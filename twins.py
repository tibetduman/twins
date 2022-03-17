import pandas as pd
import categorize as cat
from sentence_transformers import SentenceTransformer
import scipy
import time
import multiprocessing as mp

DATA = [] #usable data in numpy_array format
CATEGORY_LIST = [set() for i in range(50)] #list of categories, sets that contain company names
CATEGORY_NUM = {} #dictionary of CATEGORY_LIST set indices with category names
COMPANIES = {} #dictionary of companies by name as keys and short descriptions as values
BEST_K = 10 #the desired amount of most relevant companies
SENTENCE_LIMIT = 100 #limit on how many sentences to use BERT on
LOWEST_SENTENCE_LENGTH = 8 #lower bound on the length of a description for a sentence
model = SentenceTransformer('bert-base-nli-mean-tokens') #NLP model to use

def data_wrangling():
    """Function to handle data wrangling to make raw excel file processable."""
    print("Starting data wrangling")
    data = pd.read_excel("dataset.xls")
    data = data.drop(["uuid", "uuid_1"], axis=1)
    print(data.columns)
    data = data.to_numpy(na_value="none")
    for entry in data:
        add_company(entry[0], entry[4]) #entry[1] is short, entry[4] is long description
        #entry[2], category_list, is ignored since less comprehensive than category_group_list
        categories = seperate_by_commas_trim(entry[3])
        for category in categories:
            if category not in CATEGORY_NUM:
                category = category.lower()
                CATEGORY_NUM[category] = len(CATEGORY_NUM)
            CATEGORY_LIST[CATEGORY_NUM[category]].add(entry[0])

def add_company(name, description):
    """Function to add a company to the dictionary of companies, stores a mapping
    from name to short_description."""
    if len(description) > LOWEST_SENTENCE_LENGTH:
        COMPANIES[name] = description.lower()
    else:
        #description not long enough to be considered
        return

def get_category(company):
    """Given a company finds it's category(s) and returns them."""
    categories = []
    for category in CATEGORY_NUM:
        if company in CATEGORY_LIST[CATEGORY_NUM[category]]:
            categories.append(category)
    return categories

def seperate_by_commas_trim(sentence):
    """Function seperate a sentence by using comma as a delimiter and strips each subsentence,
    returns the resulting list of strings as a list."""
    sentence = sentence.lower()
    sentence = sentence.split(",")
    for s in sentence:
        s.strip()
    return sentence

def category_analysis():
    """Prints out useful statistics about categories of companies."""
    for category in CATEGORY_NUM:
        print("Category, {0}, has {1} many companies".format(category, len(CATEGORY_LIST[CATEGORY_NUM[category]])))
        #print_example_companies_from_category(category) #this is verbose
    print("there are {0} many companies".format(len(COMPANIES)))
    print("there are {0} many categories".format(len(CATEGORY_NUM)))

def categorize(short_description):
    """Given a short description return 3 best possible categories that best suit it."""
    category = "none"
    #TODO
    #some logic using machine learning to classify short_description into one of the categories in CATEGORY_LIST
    #currently using the model developed by SplendeourC4
    #CREDIT: https://github.com/velapartners/SplendourC4
    category = cat.solve(short_description)
    print("The best category matching the description: {0} is {1}".format(short_description, category))
    return category

def sentences_in_category(categories):
    """Given a short description search and return the best k companies in the given category.
    Where best is defined as the most relevant and close. needs update"""
    descriptions_in_category = []
    for category in categories:
        for company in CATEGORY_LIST[CATEGORY_NUM[category]]:
            descriptions_in_category.append(COMPANIES[company])
    return descriptions_in_category

def split_into_n_sublists(l, n):
    """Given a list l, splits l into n sublists of close size and returns the sublists in a list."""
    size = int(len(l) / n)
    sublists = [l[i:i + size] for i in range(0, len(l), size)]
    return sublists

def closest_k_sentences(short_description, sentences_in_category):
    """Returns the k closest companies that have the best affiliated sentences with
    the given short description."""
    corpus = model.encode(sentences_in_category)
    query = model.encode(short_description)
    distances = scipy.spatial.distance.cdist([query], corpus, "cosine")[0]
    results = zip(range(len(distances)), distances)
    print(len(sentences_in_category))
    return results

def filter_sentences(sentence_list):
    sentence_with_similarity_score = []
    for sentence in sentence_list:
        sentence_with_similarity_score.append((sentence, jaccard_similarity(sentence, short_description)))
    sentence_with_similarity_score = sorted(sentence_with_similarity_score, key=lambda x: x[1])
    return sentence_with_similarity_score

def jaccard_similarity(sentence1, sentence2):
    intersection = 0
    for word in sentence1:
        if word in sentence2:
            intersection += 1
    return intersection / (len(sentence1) + len(sentence2))

if __name__ == '__main__':
    data_wrangling()
    category_analysis()
    start = time.perf_counter()
    number_of_processors = mp.cpu_count()
    print("Number of processors:", number_of_processors)
    short_description = "a machine learning company that focuses on online payments."
    all_sentences = filter_sentences(sentences_in_category(["payments"]))
    #all_sentences = sentences_in_category(["payments"]) #uncomment if u want no filtering
    sentences = [sentence[0] for sentence in all_sentences[0:SENTENCE_LIMIT]]
    print("Will run bert on {} many sentences.".format(len(sentences)))
    pool = mp.Pool(number_of_processors)
    optimized_sentences = split_into_n_sublists(sentences, number_of_processors)
    for i in range(number_of_processors):
        results = pool.apply_async(closest_k_sentences, args=(short_description, optimized_sentences[i]))
    pool.close()
    pool.join()
    result = results.get()
    #result = closest_k_sentences(sentences) #remove after testing
    result = sorted(result, key=lambda x: x[1])
    for idx, distance in result[0:BEST_K]:
        print(sentences[idx].strip(), "(Score: {0})".format(1-distance))
    end = time.perf_counter()
    print("It took {0} seconds to process {1} many sentences.".format(end - start, len(sentences)))