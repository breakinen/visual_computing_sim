

def jaccard_similarity(string_a,string_b):
    string_a = "China|Scene|Sunny|Forest"
    string_b = "Germany|Scene|Rainy|Forest"

    string_a_tokens = string_a.split("|")
    string_b_tokens = string_b.split("|")

    intersection = list(set(string_a_tokens)&set(string_b_tokens))
    union = list(set(string_a_tokens)|set(string_b_tokens))

    jaccard_similarity = len(intersection)/union
