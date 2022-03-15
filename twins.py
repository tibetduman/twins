import pandas as pd

DATA = [] #usable data in numpy_array format
CATEGORY_LIST = [set() for i in range(50)] #list of categories, sets that contain company names
CATEGORY_NUM = {} #dictionary of CATEGORY_LIST set indices with category names
COMPANIES = {} #dictionary of companies by name as keys and short descriptions as values
BEST_K = 10 #the desired amount of most relevant companies

def data_wrangling():
    """Function to handle data wrangling to make raw excel file processable."""
    print("Starting data wrangling")
    data = pd.read_excel("dataset.xls")
    data = data.drop(["uuid", "uuid_1"], axis=1)
    data = data.to_numpy(na_value="none")
    most_categories = 0
    most_categoried_company = None
    for entry in data:
        add_company(entry[0], entry[1])
        #entry[2], category_list, is ignored since less comprehensive than category_group_list
        categories = seperate_by_commas_trim(entry[3])
        if len(categories) > most_categories:
            most_categoried_company = entry[0]
            most_categories = len(categories)
        for category in categories:
            if category not in CATEGORY_NUM:
                CATEGORY_NUM[category] = len(CATEGORY_NUM)
            CATEGORY_LIST[CATEGORY_NUM[category]].add(entry[0])
    print("Most categories for a single company is {0}  with {1} many categories".format(most_categoried_company, most_categories))

def add_company(name, short_description):
    """Function to add a company to the dictionary of companies, stores a mapping
    from name to short_description."""
    COMPANIES[name] = short_description

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

def print_example_companies_from_category(category):
    """Prints out 3 different companies from given category."""
    companies = iter(CATEGORY_LIST[CATEGORY_NUM[category]])
    for i in range(3):
        some_company = next(companies)
        print("From category {0} one example company is {1} with short description {2}".format(category, some_company, COMPANIES[some_company]))

def categorize(short_description):
    """Given a short description return 3 best possible categories that best suit it."""
    category = "none"
    #TODO
    #some logic using machine learning to classify short_description into one of the categories in CATEGORY_LIST
    print("The best category matching the description: {0} is {1}".format(short_description, category))
    return category

def find_closest_k_companies_in_category(short_description, k, category):
    """Given a short description search and return the best k companies in the given category.
    Where best is defined as the most relevant and close."""
    #TODO
    return ["Company X"] * k

def find_best_k(short_description, k):
    """Given a short description search and return the most relevant k companies."""
    return find_closest_k_companies_in_category(short_description, k, categorize(short_description))

def solve_report(short_description):
    """Given a short description solves the problem of finding closely related companies,
    and reports them appropriately."""
    solution = find_best_k(short_description, BEST_K)
    i = 1
    for company in solution:
        print("{0}th best fit is {1}, with the category(s) {2} and short description {3}".format(i, company, get_category(company), COMPANIES[company]))

data_wrangling()
#category_analysis()
